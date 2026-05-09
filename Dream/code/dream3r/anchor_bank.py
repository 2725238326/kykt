"""
AnchorBank — bounded spatial key/value memory for Dream3R C2 Memory.

Lifecycle per window tick:
  1. read():       top-k retrieval for current frame queries
  2. write():      bus-gated insertion of new anchors
  3. prune():      utility-based eviction when at capacity
  4. quarantine(): mark uncertain entries for exclusion from reads

The bank is stateful across windows but resets per sequence.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class WriteResult:
    written_mask: torch.Tensor      # [B, N_candidates] — True = accepted
    n_written: int
    n_suppressed: int
    n_quarantined: int


@dataclass
class ReadResult:
    values: torch.Tensor            # [B, Q, K, D_v]
    keys: torch.Tensor              # [B, Q, K, D_k]
    scores: torch.Tensor            # [B, Q, K]
    indices: torch.Tensor           # [B, Q, K]


@dataclass
class PruneResult:
    n_pruned: int
    lowest_utility: float


class AnchorBank(nn.Module):
    """
    Bounded spatial K/V memory bank.

    Stores up to `capacity` anchor entries, each consisting of a key vector,
    a value vector, and metadata (utility score, write confidence, timestamps).

    Bus-gated writes:
      - Suppressed if dynamic_ratio > dynamic_threshold
      - Suppressed if conflict_score > conflict_threshold
      - Quarantined if confidence is between quarantine and conflict thresholds

    Utility-based pruning preserves future-useful entries over LRU.

    Args:
        capacity:            max entries
        d_key:               key dimension
        d_value:             value dimension
        dynamic_threshold:   bus dynamic_ratio above which writes are suppressed
        conflict_threshold:  bus conflict_score above which writes are suppressed
        quarantine_threshold: conflict_score above which entries are quarantined
        utility_decay:       exponential decay factor for access-based utility
    """

    def __init__(self, capacity: int = 256, d_key: int = 128,
                 d_value: int = 128, dynamic_threshold: float = 0.7,
                 conflict_threshold: float = 0.8,
                 quarantine_threshold: float = 0.5,
                 utility_decay: float = 0.99):
        super().__init__()
        self.capacity = capacity
        self.d_key = d_key
        self.d_value = d_value
        self.dynamic_threshold = dynamic_threshold
        self.conflict_threshold = conflict_threshold
        self.quarantine_threshold = quarantine_threshold
        self.utility_decay = utility_decay

        self.register_buffer("keys", torch.zeros(1, capacity, d_key))
        self.register_buffer("values", torch.zeros(1, capacity, d_value))
        self.register_buffer("valid", torch.zeros(1, capacity, dtype=torch.bool))
        self.register_buffer("quarantined", torch.zeros(1, capacity, dtype=torch.bool))
        self.register_buffer("utility", torch.zeros(1, capacity))
        self.register_buffer("write_confidence", torch.zeros(1, capacity))
        self.register_buffer("access_count", torch.zeros(1, capacity, dtype=torch.long))
        self.register_buffer("write_timestep", torch.zeros(1, capacity, dtype=torch.long))
        self.register_buffer("last_access_timestep", torch.zeros(1, capacity, dtype=torch.long))
        self.register_buffer("_write_cursor", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_current_timestep", torch.zeros(1, dtype=torch.long))

        self.key_proj = nn.Linear(d_key, d_key, bias=False)
        self.utility_scorer = nn.Sequential(
            nn.Linear(d_key + d_value + 3, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def reset(self, batch_size: int = 1, device: Optional[torch.device] = None):
        dev = device or self.keys.device
        self.keys = torch.zeros(batch_size, self.capacity, self.d_key, device=dev)
        self.values = torch.zeros(batch_size, self.capacity, self.d_value, device=dev)
        self.valid = torch.zeros(batch_size, self.capacity, dtype=torch.bool, device=dev)
        self.quarantined = torch.zeros(batch_size, self.capacity, dtype=torch.bool, device=dev)
        self.utility = torch.zeros(batch_size, self.capacity, device=dev)
        self.write_confidence = torch.zeros(batch_size, self.capacity, device=dev)
        self.access_count = torch.zeros(batch_size, self.capacity, dtype=torch.long, device=dev)
        self.write_timestep = torch.zeros(batch_size, self.capacity, dtype=torch.long, device=dev)
        self.last_access_timestep = torch.zeros(batch_size, self.capacity, dtype=torch.long, device=dev)
        self._write_cursor = torch.zeros(batch_size, dtype=torch.long, device=dev)
        self._current_timestep = torch.zeros(batch_size, dtype=torch.long, device=dev)

    @property
    def occupancy(self) -> torch.Tensor:
        return self.valid.sum(dim=-1).float()

    @property
    def readable_mask(self) -> torch.Tensor:
        return self.valid & ~self.quarantined

    def read(self, queries: torch.Tensor,
             top_k: int = 8) -> ReadResult:
        """
        Top-k retrieval from non-quarantined valid entries.

        Args:
            queries: [B, Q, D_k]
            top_k:   number of entries to retrieve per query
        Returns:
            ReadResult with values, keys, scores, indices
        """
        B, Q, D = queries.shape
        K = min(top_k, self.capacity)

        q = self.key_proj(queries)
        scores = torch.bmm(q, self.keys.transpose(1, 2)) / (D ** 0.5)

        mask = self.readable_mask.unsqueeze(1).expand_as(scores)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        topk_scores, topk_idx = scores.topk(K, dim=-1)

        flat_idx = topk_idx.reshape(B, -1)
        sel_keys = self.keys.gather(1, flat_idx.unsqueeze(-1).expand(-1, -1, self.d_key))
        sel_vals = self.values.gather(1, flat_idx.unsqueeze(-1).expand(-1, -1, self.d_value))
        sel_keys = sel_keys.view(B, Q, K, self.d_key)
        sel_vals = sel_vals.view(B, Q, K, self.d_value)

        unique_per_batch = topk_idx.reshape(B, -1)
        self.access_count.scatter_add_(1, unique_per_batch,
                                        torch.ones_like(unique_per_batch, dtype=torch.long))
        ts = self._current_timestep.unsqueeze(-1).expand_as(unique_per_batch)
        self.last_access_timestep.scatter_(1, unique_per_batch, ts)

        return ReadResult(
            values=sel_vals,
            keys=sel_keys,
            scores=topk_scores,
            indices=topk_idx,
        )

    def write(self, keys: torch.Tensor, values: torch.Tensor,
              confidence: Optional[torch.Tensor] = None,
              bus_dynamic_ratio: Optional[torch.Tensor] = None,
              bus_conflict_score: Optional[torch.Tensor] = None,
              ) -> WriteResult:
        """
        Bus-gated insertion of new anchor entries.

        Args:
            keys:               [B, N, D_k]
            values:             [B, N, D_v]
            confidence:         [B, N] or None — per-entry write confidence
            bus_dynamic_ratio:  [B, 1] or None
            bus_conflict_score: [B, 1] or None
        Returns:
            WriteResult with write stats
        """
        B, N, _ = keys.shape
        device = keys.device

        accept = torch.ones(B, N, dtype=torch.bool, device=device)
        quarantine_new = torch.zeros(B, N, dtype=torch.bool, device=device)

        if bus_dynamic_ratio is not None:
            suppress_dyn = (bus_dynamic_ratio > self.dynamic_threshold).squeeze(-1)
            accept = accept & ~suppress_dyn.unsqueeze(-1).expand(B, N)

        if bus_conflict_score is not None:
            suppress_conf = (bus_conflict_score > self.conflict_threshold).squeeze(-1)
            accept = accept & ~suppress_conf.unsqueeze(-1).expand(B, N)

            quar = (bus_conflict_score > self.quarantine_threshold).squeeze(-1)
            quar = quar & ~suppress_conf
            quarantine_new = quarantine_new | quar.unsqueeze(-1).expand(B, N)

        n_written = 0
        n_suppressed = 0
        n_quarantined = 0

        if confidence is None:
            confidence = torch.ones(B, N, device=device)

        n_accepted = accept.sum().item()
        n_suppressed = B * N - n_accepted
        n_quarantined = (accept & quarantine_new).sum().item()

        for b in range(B):
            accepted_idx = accept[b].nonzero(as_tuple=True)[0]
            if accepted_idx.numel() == 0:
                continue

            n_to_write = accepted_idx.numel()
            free_slots = (~self.valid[b]).nonzero(as_tuple=True)[0]

            if free_slots.numel() >= n_to_write:
                write_positions = free_slots[:n_to_write]
            else:
                n_from_free = free_slots.numel()
                n_evict = n_to_write - n_from_free
                _, evict_order = self.utility[b].topk(n_evict, largest=False)
                write_positions = torch.cat([free_slots, evict_order]) if n_from_free > 0 else evict_order

            self.keys[b, write_positions] = keys[b, accepted_idx].float()
            self.values[b, write_positions] = values[b, accepted_idx].float()
            self.valid[b, write_positions] = True
            self.write_confidence[b, write_positions] = confidence[b, accepted_idx].float()
            self.write_timestep[b, write_positions] = self._current_timestep[b]
            self.last_access_timestep[b, write_positions] = self._current_timestep[b]
            self.access_count[b, write_positions] = 0
            self.utility[b, write_positions] = confidence[b, accepted_idx].float()

            quar_mask = quarantine_new[b, accepted_idx]
            self.quarantined[b, write_positions] = quar_mask

            if write_positions.numel() > 0:
                self._write_cursor[b] = (write_positions[-1] + 1) % self.capacity

        return WriteResult(
            written_mask=accept,
            n_written=n_accepted,
            n_suppressed=n_suppressed,
            n_quarantined=n_quarantined,
        )

    def prune(self, keep_ratio: float = 0.75) -> PruneResult:
        """
        Utility-based pruning: evict lowest-utility entries to reach keep_ratio.

        Utility is a learned function of (key, value, access_count,
        recency, write_confidence).
        """
        B = self.keys.shape[0]
        target = int(self.capacity * keep_ratio)
        total_pruned = 0
        min_util = float("inf")

        for b in range(B):
            valid_idx = self.valid[b].nonzero(as_tuple=True)[0]
            n_valid = valid_idx.shape[0]
            if n_valid <= target:
                continue

            recency = (self._current_timestep[b] - self.last_access_timestep[b, valid_idx]).float()
            recency = recency / (recency.max() + 1e-8)
            access = self.access_count[b, valid_idx].float()
            access = access / (access.max() + 1e-8)
            conf = self.write_confidence[b, valid_idx]

            meta = torch.stack([access, recency, conf], dim=-1)
            kv = torch.cat([self.keys[b, valid_idx], self.values[b, valid_idx]], dim=-1)
            feat = torch.cat([kv, meta], dim=-1)

            with torch.no_grad():
                scores = self.utility_scorer(feat).squeeze(-1)

            self.utility[b, valid_idx] = scores

            n_to_prune = n_valid - target
            _, prune_order = scores.topk(n_to_prune, largest=False)
            prune_idx = valid_idx[prune_order]

            self.valid[b, prune_idx] = False
            self.quarantined[b, prune_idx] = False
            total_pruned += n_to_prune

            if scores.numel() > 0:
                min_util = min(min_util, scores.min().item())

        return PruneResult(
            n_pruned=total_pruned,
            lowest_utility=min_util if min_util != float("inf") else 0.0,
        )

    def quarantine(self, indices: torch.Tensor):
        """
        Mark entries as quarantined (excluded from reads until validated).

        Args:
            indices: [B, N] — entry indices to quarantine
        """
        B = indices.shape[0]
        for b in range(B):
            idx = indices[b]
            valid_mask = (idx >= 0) & (idx < self.capacity)
            self.quarantined[b, idx[valid_mask]] = True

    def unquarantine(self, indices: torch.Tensor):
        """Release quarantined entries back to readable pool."""
        B = indices.shape[0]
        for b in range(B):
            idx = indices[b]
            valid_mask = (idx >= 0) & (idx < self.capacity)
            self.quarantined[b, idx[valid_mask]] = False

    def tick(self):
        self._current_timestep += 1
        self.utility *= self.utility_decay

    def state_snapshot(self) -> Dict[str, torch.Tensor]:
        return {
            "keys": self.keys.clone(),
            "values": self.values.clone(),
            "valid": self.valid.clone(),
            "quarantined": self.quarantined.clone(),
            "utility": self.utility.clone(),
            "occupancy": self.occupancy,
        }
