"""
Dream3R computational cores C1-C5.

v0.2 modules (MemorySSM_v01, Composer_v01) are preserved for ablation.
v0.3 modules (SpatialMemory, ComposerRouter) are the current defaults.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List

from dream3r.nsa_attention import NSAAttention
from dream3r.anchor_bank import AnchorBank
from dream3r.composer_experts import ExpertRegistry
from dream3r.composer_experts.base_adapter import ExpertOutput


# ---------------------------------------------------------------------------
# C1: Perceiver
# ---------------------------------------------------------------------------

class Perceiver(nn.Module):
    """
    Per-frame ViT backbone → T1 frame tokens, T2 pointmap/confidence,
    T3 evidence signals (17 named signals, each with independent projector).
    """

    EVIDENCE_SIGNALS = [
        "pose_novelty", "view_overlap", "reprojection_residual",
        "pointmap_conflict", "confidence_drop", "latent_drift_proxy",
        "dynamic_ratio", "optical_flow_conflict", "object_track_stability",
        "loop_candidate_score", "anchor_importance", "cache_pressure",
        "external_memory_overlap", "prior_rgb_conflict",
        "blur_or_low_light_score", "uncertainty_area",
        "model_capability_match",
    ]

    def __init__(self, d_model: int = 768, n_evidence: int = 17,
                 d_evidence: int = 32, img_size: int = 224,
                 patch_size: int = 16, use_backbone: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_evidence = n_evidence
        self.d_evidence = d_evidence
        self.use_backbone = use_backbone

        if use_backbone:
            try:
                import timm
                self.backbone = timm.create_model(
                    "vit_base_patch16_224", pretrained=False,
                    num_classes=0, global_pool="",
                )
                backbone_dim = 768
            except ImportError:
                self.backbone = None
                backbone_dim = d_model
            self.backbone_proj = nn.Linear(backbone_dim, d_model) if backbone_dim != d_model else nn.Identity()
        else:
            self.backbone = None
            self.backbone_proj = nn.Identity()

        self.pointmap_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
        self.evidence_projectors = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(d_model, d_model // 4), nn.GELU(),
                nn.Linear(d_model // 4, d_evidence),
            )
            for name in self.EVIDENCE_SIGNALS
        })

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        # [B, N, 3, H, W] -> [B, N, P, D]
        B, N = images.shape[:2]
        flat = images.reshape(B * N, *images.shape[2:])
        features = self.backbone(flat)
        features = self.backbone_proj(features)
        return features.view(B, N, features.shape[1], features.shape[2])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, N, 3, H, W] raw images OR [B, N, P, D] pre-extracted
        Returns:
            t1, t2_pointmap, t2_confidence, t3_evidence, t3_named, perception_summary
        """
        if self.use_backbone and self.backbone is not None and x.dim() == 5:
            t1 = self.encode_images(x)
        else:
            t1 = x

        t2_pointmap = self.pointmap_head(t1)              # [B, N, P, 3]
        t2_confidence = torch.sigmoid(self.confidence_head(t1))  # [B, N, P, 1]

        pooled = t1.mean(dim=2)  # [B, N, D]

        t3_named = {}
        t3_list = []
        for name in self.EVIDENCE_SIGNALS:
            sig = self.evidence_projectors[name](pooled)  # [B, N, d_ev]
            t3_named[name] = sig
            t3_list.append(sig)

        t3 = torch.stack(t3_list, dim=2)  # [B, N, 17, d_ev]
        perception_summary = t1.mean(dim=(1, 2))  # [B, D]

        return {
            "t1": t1,
            "t2_pointmap": t2_pointmap,
            "t2_confidence": t2_confidence,
            "t3_evidence": t3,
            "t3_named": t3_named,
            "perception_summary": perception_summary,
        }


# ---------------------------------------------------------------------------
# C2: Executive Memory — A1 branching + bus-informed decisions
# ---------------------------------------------------------------------------

class MemorySSM_v01(nn.Module):
    """
    [v0.1 — preserved for ablation ABL-v02-1]
    Recurrent state controller with 5 distinct update modes.

    A1 update modes:
      0 = full_update:     standard GRU forward, full gain
      1 = pose_adaptive:   gain scaled by pose_novelty evidence
      2 = kalman:          residual-weighted update (new - old, scaled by confidence)
      3 = skip:            copy prev_state unchanged
      4 = reset:           zero state

    A2: write_decision conditioned on suppress_mask (CR-2).
    Bus reads: dynamic_ratio (Permanence), conflict_score (Critic t-1).
    """

    # Named indices for update_kind
    FULL, POSE_ADAPTIVE, KALMAN, SKIP, RESET = 0, 1, 2, 3, 4

    def __init__(self, d_percept: int = 768, d_evidence_flat: int = 544,
                 d_state: int = 256, n_layers: int = 6,
                 d_bus_context: int = 3):
        super().__init__()
        self.d_state = d_state
        # d_bus_context: dynamic_ratio(1) + conflict_score(1) + drift(1)
        d_input = d_percept + d_evidence_flat + d_bus_context
        self.input_proj = nn.Linear(d_input, d_state)
        self.layers = nn.ModuleList([
            nn.GRUCell(d_state, d_state) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_state)

        # A1 heads
        self.update_classifier = nn.Linear(d_state, 5)
        self.full_gate = nn.Linear(d_state, 1)
        self.pose_gate = nn.Linear(d_state, 1)
        self.kalman_gain = nn.Linear(d_state, d_state)

        # A2 write head
        self.write_head = nn.Linear(d_state, 4)
        # A3 anchor scorer
        self.anchor_scorer = nn.Linear(d_state, 1)
        # Drift proxy
        self.drift_proj = nn.Linear(d_state, 1)

    def init_state(self, B: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(B, self.d_state, device=device)

    def forward(self, perception_summary: torch.Tensor,
                evidence_flat: torch.Tensor,
                prev_state: torch.Tensor,
                suppress_mask: Optional[torch.Tensor] = None,
                bus_dynamic_ratio: Optional[torch.Tensor] = None,
                bus_conflict_score: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            perception_summary: [B, d_percept]
            evidence_flat:      [B, d_evidence_flat]
            prev_state:         [B, d_state]
            suppress_mask:      [B] CR-2 from Permanence
            bus_dynamic_ratio:  [B, 1] from bus (Permanence)
            bus_conflict_score: [B, 1] from bus (Critic t-1)
        """
        B = perception_summary.shape[0]
        device = perception_summary.device

        # Pack bus context (null protocol: zeros if not available)
        dyn = bus_dynamic_ratio if bus_dynamic_ratio is not None else torch.zeros(B, 1, device=device)
        conf = bus_conflict_score if bus_conflict_score is not None else torch.zeros(B, 1, device=device)
        prev_drift = self.drift_proj(prev_state).detach()  # [B, 1] don't backprop through prev drift read
        bus_ctx = torch.cat([dyn, conf, prev_drift], dim=-1)  # [B, 3]

        x = self.input_proj(torch.cat([perception_summary, evidence_flat, bus_ctx], dim=-1))

        # GRU stack → candidate new state
        h = prev_state
        for gru in self.layers:
            h = gru(x, h)
        h = self.norm(h)

        # --- A1: classify then branch ---
        update_logits = self.update_classifier(h)  # [B, 5]
        update_probs = F.softmax(update_logits, dim=-1)  # [B, 5]

        # Compute each update mode's output
        g_full = torch.sigmoid(self.full_gate(h))           # [B, 1]
        g_pose = torch.sigmoid(self.pose_gate(h))           # [B, 1]
        k_gain = torch.sigmoid(self.kalman_gain(h))         # [B, d_state]
        residual = h - prev_state                           # [B, d_state]

        s_full = prev_state + g_full * residual
        s_pose = prev_state + (g_full * g_pose) * residual
        s_kalman = prev_state + k_gain * residual
        s_skip = prev_state
        s_reset = torch.zeros_like(prev_state)

        # Soft mixture (differentiable; hard argmax at inference)
        modes = torch.stack([s_full, s_pose, s_kalman, s_skip, s_reset], dim=1)  # [B, 5, d]
        new_state = (update_probs.unsqueeze(-1) * modes).sum(dim=1)  # [B, d]

        # --- A2: write decision (CR-2 gated) ---
        write_logits = self.write_head(new_state)  # [B, 4]
        if suppress_mask is not None:
            write_logits = write_logits.clone()
            write_logits[:, 0] = write_logits[:, 0] - 1e9 * suppress_mask

        # --- A3: anchor scoring ---
        anchor_scores = self.anchor_scorer(new_state)  # [B, 1]

        # Drift proxy for bus
        drift = self.drift_proj(new_state)  # [B, 1]

        return {
            "latent_state": new_state,
            "update_kind": update_logits,
            "update_probs": update_probs,
            "write_decision": write_logits,
            "anchor_scores": anchor_scores,
            "latent_drift_proxy": drift,
        }


# ---------------------------------------------------------------------------
# C3: Permanence — Slot Attention + bus-informed mint control
# ---------------------------------------------------------------------------

class Permanence(nn.Module):
    """
    Object identity tracking via Slot Attention.
    A6: region_decision (suppress/admit/defer), dynamic_ratio, suppress handoff.
    Bus reads: conflict_score → modulates object mint conservatism.
    """

    def __init__(self, d_input: int = 768, d_slot: int = 128,
                 n_slots: int = 16, n_iters: int = 3):
        super().__init__()
        self.d_slot = d_slot
        self.n_slots = n_slots
        self.n_iters = n_iters

        self.slot_mu = nn.Parameter(torch.randn(1, 1, d_slot) * (d_slot ** -0.5))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, d_slot))

        self.input_norm = nn.LayerNorm(d_input)
        self.input_proj = nn.Linear(d_input, d_slot)

        self.slot_norm = nn.LayerNorm(d_slot)
        self.k_proj = nn.Linear(d_slot, d_slot, bias=False)
        self.q_proj = nn.Linear(d_slot, d_slot, bias=False)
        self.v_proj = nn.Linear(d_slot, d_slot, bias=False)

        self.gru = nn.GRUCell(d_slot, d_slot)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_slot),
            nn.Linear(d_slot, d_slot * 2),
            nn.ReLU(),
            nn.Linear(d_slot * 2, d_slot),
        )

        # A6 heads
        self.region_head = nn.Linear(d_slot, 3)
        self.dynamic_head = nn.Linear(d_slot, 1)
        # Conflict-aware mint gate: when conflict is high, be more conservative
        self.mint_gate = nn.Linear(d_slot + 1, 1)  # slot feature + conflict_score

    def _init_slots(self, B: int, device: torch.device) -> torch.Tensor:
        mu = self.slot_mu.expand(B, self.n_slots, -1)
        sigma = self.slot_log_sigma.exp().expand(B, self.n_slots, -1)
        return mu + sigma * torch.randn_like(mu)

    def forward(self, features: torch.Tensor,
                prev_slots: Optional[torch.Tensor] = None,
                bus_conflict_score: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features:           [B, N, d_input]
            prev_slots:         [B, n_slots, d_slot] or None
            bus_conflict_score: [B, 1] from bus (Critic t-1)
        """
        B, N, _ = features.shape
        device = features.device
        inputs = self.input_proj(self.input_norm(features))
        k = self.k_proj(inputs)
        v = self.v_proj(inputs)

        slots = prev_slots if prev_slots is not None else self._init_slots(B, device)

        for _ in range(self.n_iters):
            slots_prev = slots
            q = self.q_proj(self.slot_norm(slots))
            attn_logits = torch.einsum("bsd,bnd->bsn", q, k) * (self.d_slot ** -0.5)
            attn = F.softmax(attn_logits, dim=1)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
            updates = torch.einsum("bsn,bnd->bsd", attn, v)
            slots = self.gru(
                updates.reshape(B * self.n_slots, self.d_slot),
                slots_prev.reshape(B * self.n_slots, self.d_slot),
            ).view(B, self.n_slots, self.d_slot)
            slots = slots + self.mlp(slots)

        region_logits = self.region_head(slots)  # [B, S, 3]
        dynamic_ratio = torch.sigmoid(self.dynamic_head(slots.mean(dim=1)))  # [B, 1]

        # Mint gate: modulated by conflict_score from bus
        conflict = bus_conflict_score if bus_conflict_score is not None else torch.zeros(B, 1, device=device)
        conflict_expanded = conflict.unsqueeze(1).expand(B, self.n_slots, 1)
        mint_input = torch.cat([slots, conflict_expanded], dim=-1)  # [B, S, d_slot+1]
        mint_confidence = torch.sigmoid(self.mint_gate(mint_input)).squeeze(-1)  # [B, S]

        suppress = (region_logits.argmax(dim=-1) == 0).any(dim=-1).float()

        return {
            "object_track_set": slots,
            "region_logits": region_logits,
            "dynamic_ratio": dynamic_ratio,
            "suppress_static_write": suppress,
            "mint_confidence": mint_confidence,
        }


# ---------------------------------------------------------------------------
# C4: Critic — transformer over evidence sequence
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    """
    Small transformer reading the 17 evidence tokens.
    A4: conflict_score (scalar).
    A5: repair_action (6-way, CR-1 gated on reroute).
    """
    REROUTE_IDX = 2

    def __init__(self, n_evidence: int = 17, d_evidence: int = 32,
                 d_critic: int = 256, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.token_proj = nn.Linear(d_evidence, d_critic)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_critic, nhead=n_heads, dim_feedforward=d_critic * 2,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.conflict_head = nn.Linear(d_critic, 1)
        self.repair_head = nn.Linear(d_critic, 6)

    def forward(self, evidence: torch.Tensor,
                cr1_mask: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            evidence: [B, n_evidence, d_evidence]
            cr1_mask: scalar or [B], 1=allow reroute, 0=block
        """
        x = self.token_proj(evidence)
        x = self.encoder(x)
        pooled = x.mean(dim=1)

        conflict = self.conflict_head(pooled)
        repair = self.repair_head(pooled)

        if cr1_mask is not None:
            cr1 = cr1_mask.float()
            if cr1.dim() == 0:
                cr1 = cr1.unsqueeze(0)
            cr1 = cr1.expand(repair.shape[0])
            repair = repair.clone()
            repair[:, self.REROUTE_IDX] = torch.where(
                cr1 > 0.5, repair[:, self.REROUTE_IDX],
                torch.full_like(repair[:, self.REROUTE_IDX], -65000),
            )

        return {
            "conflict_score": conflict,
            "repair_logits": repair,
            "recommended_action": repair.argmax(dim=-1),
        }


# ---------------------------------------------------------------------------
# C5: Composer — parameter-free table join
# ---------------------------------------------------------------------------

class Composer_v01(nn.Module):
    """[v0.1 — preserved for ablation] Zero parameters. regime @ capability_cards.T -> match -> rank."""

    def __init__(self, n_regimes: int = 5, n_models: int = 8):
        super().__init__()
        self.n_regimes = n_regimes
        self.n_models = n_models
        self.register_buffer("capability_cards",
                             torch.ones(n_models, n_regimes) / n_regimes)

    def set_capability_cards(self, cards: torch.Tensor):
        self.capability_cards.copy_(cards)

    def forward(self, regime_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        match = regime_probs @ self.capability_cards.t()
        scores, indices = match.sort(dim=-1, descending=True)
        regret = scores[:, 0] - scores[:, 1] if self.n_models > 1 else match.new_zeros(match.shape[0])
        return {
            "capability_match": match,
            "route_recommendation": indices,
            "route_regret": regret,
        }


# Aliases for backward compatibility
MemorySSM = MemorySSM_v01
Composer = Composer_v01


# ---------------------------------------------------------------------------
# C2 v0.3: Spatial Memory — NSA + AnchorBank + latent state recurrence
# ---------------------------------------------------------------------------

class StateTokenRecurrence(nn.Module):
    """CUT3R-style latent state token update via cross-attention with frame tokens."""

    def __init__(self, d_model: int, n_state_tokens: int, n_heads: int = 4):
        super().__init__()
        self.n_state_tokens = n_state_tokens
        self.init_tokens = nn.Parameter(
            torch.randn(1, n_state_tokens, d_model) * (d_model ** -0.5)
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True,
        )
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.init_tokens.expand(batch_size, -1, -1).clone()

    def forward(self, prev_state: torch.Tensor,
                frame_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prev_state:   [B, S, D] — previous latent state tokens
            frame_tokens: [B, P, D] — current frame tokens
        Returns:
            new_state: [B, S, D]
        """
        h = self.norm1(prev_state)
        h = prev_state + self.cross_attn(h, frame_tokens, frame_tokens)[0]

        h2 = self.norm2(h)
        h = h + self.self_attn(h2, h2, h2)[0]

        h = h + self.ffn(self.norm3(h))
        return h


class SpatialMemory(nn.Module):
    """
    C2 v0.3: NSA-backed spatial memory with AnchorBank and latent recurrence.

    Architecture:
      1. Latent state tokens recur via cross-attention with frame tokens (compressed branch source)
      2. AnchorBank stores/retrieves spatial K/V anchors (selected branch source)
      3. Sliding window buffer holds recent frames (sliding branch source)
      4. NSA fuses all three branches
      5. Bus-gated writes to AnchorBank

    Args:
        d_model:          token dimension for all internal representations
        n_state_tokens:   number of latent state tokens (CUT3R-style)
        bank_capacity:    max AnchorBank entries
        nsa_n_select_k:   top-k for NSA selected branch
        nsa_n_heads:      attention heads in NSA
        sliding_window:   number of recent windows to keep in sliding buffer
        n_evidence:       number of evidence signals (for write decision)
        d_evidence:       evidence signal dimension
    """

    def __init__(self, d_model: int = 128, n_state_tokens: int = 32,
                 bank_capacity: int = 256, nsa_n_select_k: int = 8,
                 nsa_n_heads: int = 4, sliding_window: int = 4,
                 n_evidence: int = 17, d_evidence: int = 32):
        super().__init__()
        self.d_model = d_model
        self.n_state_tokens = n_state_tokens
        self.sliding_window = sliding_window

        self.state_recurrence = StateTokenRecurrence(
            d_model, n_state_tokens, n_heads=nsa_n_heads,
        )

        self.anchor_bank = AnchorBank(
            capacity=bank_capacity, d_key=d_model, d_value=d_model,
        )

        self.nsa = NSAAttention(
            d_model=d_model, n_compress=n_state_tokens,
            n_select_k=nsa_n_select_k, n_heads=nsa_n_heads,
        )

        self.frame_proj = nn.Linear(768, d_model)
        self.evidence_proj = nn.Linear(n_evidence * d_evidence, d_model)

        self.write_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        self.key_gen = nn.Linear(d_model, d_model)
        self.value_gen = nn.Linear(d_model, d_model)

        self.drift_proj = nn.Linear(d_model, 1)

        self.update_classifier = nn.Linear(d_model, 5)
        self.write_head = nn.Linear(d_model, 4)
        self.anchor_scorer = nn.Linear(d_model, 1)

        self._sliding_buffer: List[torch.Tensor] = []

    def init_state(self, batch_size: int,
                   device: torch.device) -> torch.Tensor:
        self.anchor_bank.reset(batch_size, device)
        self._sliding_buffer = []
        return self.state_recurrence.init_state(batch_size, device)

    def forward(self, frame_tokens: torch.Tensor,
                evidence_flat: torch.Tensor,
                prev_state_tokens: torch.Tensor,
                bus_dynamic_ratio: Optional[torch.Tensor] = None,
                bus_conflict_score: Optional[torch.Tensor] = None,
                suppress_mask: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            frame_tokens:      [B, P, D_frame] (D_frame=768)
            evidence_flat:     [B, n_ev * d_ev]
            prev_state_tokens: [B, S, D] latent state from previous window
            bus_dynamic_ratio: [B, 1] from Permanence
            bus_conflict_score:[B, 1] from Critic t-1
            suppress_mask:     [B] CR-2 from Permanence
        Returns:
            latent_state_tokens, update_kind, write_decision, retrieval_log, etc.
        """
        B = frame_tokens.shape[0]
        device = frame_tokens.device

        frame_proj = self.frame_proj(frame_tokens)
        evidence_proj = self.evidence_proj(evidence_flat).unsqueeze(1)

        new_state = self.state_recurrence(prev_state_tokens, frame_proj)

        self._sliding_buffer.append(frame_proj.detach())
        if len(self._sliding_buffer) > self.sliding_window:
            self._sliding_buffer = self._sliding_buffer[-self.sliding_window:]
        sliding = torch.cat(self._sliding_buffer, dim=1)

        bank_keys = self.anchor_bank.keys[:B]
        bank_values = self.anchor_bank.values[:B]
        bank_mask = self.anchor_bank.readable_mask[:B]

        nsa_out = self.nsa(
            query=frame_proj,
            compressed_ctx=new_state,
            bank_keys=bank_keys,
            bank_values=bank_values,
            sliding_buffer=sliding,
            bank_mask=bank_mask,
        )

        memory_output = nsa_out["output"]
        pooled = memory_output.mean(dim=1)

        update_logits = self.update_classifier(pooled)
        update_probs = F.softmax(update_logits, dim=-1)

        write_logits = self.write_head(pooled)
        if suppress_mask is not None:
            write_logits = write_logits.clone()
            write_logits[:, 0] = write_logits[:, 0] - 1e9 * suppress_mask

        write_conf = torch.sigmoid(self.write_gate(
            torch.cat([pooled, evidence_proj.squeeze(1)], dim=-1)
        ))

        n_write_candidates = min(8, frame_proj.shape[1])
        anchor_scores_per_token = torch.bmm(
            self.key_gen(frame_proj),
            pooled.unsqueeze(-1)
        ).squeeze(-1)
        _, top_indices = anchor_scores_per_token.topk(n_write_candidates, dim=1)
        write_keys = self.key_gen(frame_proj).gather(
            1, top_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
        )
        write_values = self.value_gen(memory_output).gather(
            1, top_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
        )

        write_result = self.anchor_bank.write(
            keys=write_keys,
            values=write_values,
            confidence=write_conf.expand(B, n_write_candidates),
            bus_dynamic_ratio=bus_dynamic_ratio,
            bus_conflict_score=bus_conflict_score,
        )
        self.anchor_bank.tick()

        anchor_scores = self.anchor_scorer(pooled)
        drift = self.drift_proj(pooled)

        return {
            "latent_state_tokens": new_state,
            "latent_state": pooled,
            "update_kind": update_logits,
            "update_probs": update_probs,
            "write_decision": write_logits,
            "anchor_scores": anchor_scores,
            "latent_drift_proxy": drift,
            "nsa_branch_weights": nsa_out["branch_weights"],
            "nsa_selected_indices": nsa_out["selected_indices"],
            "bank_occupancy": self.anchor_bank.occupancy,
            "write_result": {
                "n_written": write_result.n_written,
                "n_suppressed": write_result.n_suppressed,
                "n_quarantined": write_result.n_quarantined,
            },
        }


# ---------------------------------------------------------------------------
# C5 v0.3: Composer Router — expert dispatch with cost-normalized matching
# ---------------------------------------------------------------------------

class ComposerRouter(nn.Module):
    """
    C5 v0.3: Expert routing with cost-normalized capability matching.

    Replaces the zero-param table join with learned routing that accounts
    for latency cost and Critic confidence feedback.

    Args:
        n_regimes:       number of reconstruction regimes
        d_routing:       routing embedding dimension
        cost_alpha:      latency cost weight (0 = ignore latency, 1 = latency-only)
        expert_registry: ExpertRegistry instance (optional, for dispatch)
    """

    def __init__(self, n_regimes: int = 5, d_routing: int = 64,
                 cost_alpha: float = 0.5,
                 expert_registry: Optional[ExpertRegistry] = None):
        super().__init__()
        self.n_regimes = n_regimes
        self.d_routing = d_routing
        self.cost_alpha = cost_alpha
        self.registry = expert_registry

        n_experts = 7
        self.register_buffer(
            "capability_cards",
            torch.ones(n_experts, n_regimes) / n_regimes,
        )
        self.register_buffer(
            "latency_costs",
            torch.ones(n_experts) * 30.0,
        )

        self.regime_encoder = nn.Sequential(
            nn.Linear(n_regimes, d_routing),
            nn.GELU(),
            nn.Linear(d_routing, d_routing),
        )
        self.confidence_gate = nn.Linear(1, d_routing)
        self.routing_head = nn.Linear(d_routing, n_experts)

    def set_capability_cards(self, cards: torch.Tensor):
        self.capability_cards.copy_(cards)

    def set_latency_costs(self, costs: torch.Tensor):
        self.latency_costs.copy_(costs)

    def load_from_registry(self):
        if self.registry is None:
            return
        cards = self.registry.capability_matrix()
        latencies = self.registry.latency_vector()
        n = min(cards.shape[0], self.capability_cards.shape[0])
        self.capability_cards[:n] = cards[:n]
        self.latency_costs[:n] = latencies[:n]

    def forward(self, regime_probs: torch.Tensor,
                critic_confidence: Optional[torch.Tensor] = None,
                latency_budget_ms: Optional[float] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            regime_probs:      [B, n_regimes]
            critic_confidence: [B, 1] or None
            latency_budget_ms: float or None
        Returns:
            capability_match, route_recommendation, route_regret,
            routing_logits, selected_expert
        """
        B = regime_probs.shape[0]
        device = regime_probs.device

        table_match = regime_probs @ self.capability_cards.t()

        latency_norm = self.latency_costs / (self.latency_costs.max() + 1e-8)
        cost_penalty = self.cost_alpha * latency_norm.unsqueeze(0).expand(B, -1)

        if latency_budget_ms is not None:
            over_budget = (self.latency_costs > latency_budget_ms).float()
            cost_penalty = cost_penalty + over_budget.unsqueeze(0) * 10.0

        regime_embed = self.regime_encoder(regime_probs)
        if critic_confidence is not None:
            conf_mod = self.confidence_gate(critic_confidence)
            regime_embed = regime_embed + conf_mod

        learned_logits = self.routing_head(regime_embed)
        combined = table_match + 0.1 * learned_logits - cost_penalty

        scores, indices = combined.sort(dim=-1, descending=True)
        regret = scores[:, 0] - scores[:, 1] if combined.shape[1] > 1 else combined.new_zeros(B)
        selected = indices[:, 0]

        return {
            "capability_match": table_match,
            "route_recommendation": indices,
            "route_regret": regret,
            "routing_logits": learned_logits,
            "cost_adjusted_scores": combined,
            "selected_expert": selected,
        }

    def dispatch(self, expert_id: int, images: torch.Tensor,
                 context: Optional[Dict[str, torch.Tensor]] = None,
                 ) -> Optional[ExpertOutput]:
        if self.registry is None:
            return None
        names = sorted(self.registry.names)
        if expert_id >= len(names):
            return None
        adapter = self.registry.get(names[expert_id])
        return adapter.forward(images, context)
