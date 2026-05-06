"""
Cross-spec Memory Bus (C6) — the typed tensor namespace connecting all modules.

Implements three surfaces:
  1. Published signals (read-only contract)
  2. Handoff signals (binding commands)
  3. CR-1..CR-6 gate modules
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from enum import IntEnum


class EvidenceLabel(IntEnum):
    UNKNOWN = 0
    SPECULATIVE = 1
    INFERRED = 2
    PAPER_DERIVED = 3
    PAPER_PROVEN = 4


@dataclass
class BusSignal:
    tensor: torch.Tensor
    label: EvidenceLabel
    producer: str
    timestep: int


class MemoryBus(nn.Module):
    """
    C6: Cross-spec memory bus. Zero learned parameters.
    Implements publish/read/handoff + CR-1..CR-6 gates.
    """

    def __init__(self, epsilon_spread: float = 0.05, epsilon_tie: float = 0.05):
        super().__init__()
        self.epsilon_spread = epsilon_spread
        self.epsilon_tie = epsilon_tie
        self._signals: Dict[str, BusSignal] = {}
        self._handoffs: Dict[str, BusSignal] = {}
        self._contract_log: list = []
        self._owner_table = {
            "conflict_score": "critic",
            "recommended_action": "critic",
            "route_history": "critic",
            "route_regret_estimate": "critic",
            "latent_drift_proxy": "memory",
            "anchor_set": "memory",
            "policy_log": "memory",
            "dynamic_ratio": "permanence",
            "object_track_stability": "permanence",
            "suppress_static_write": "permanence",
            "admit_static_write": "permanence",
            "capability_match": "composer",
            "route_recommendation": "composer",
            "route_regret": "composer",
            "regime_card": "composer",
        }

    def reset(self):
        self._signals.clear()
        self._handoffs.clear()

    def publish(self, signal_name: str, tensor: torch.Tensor,
                label: EvidenceLabel, producer: str, timestep: int):
        expected_owner = self._owner_table.get(signal_name)
        if expected_owner and expected_owner != producer:
            raise RuntimeError(
                f"Contract violation: {producer} tried to publish {signal_name}, "
                f"owned by {expected_owner}"
            )
        self._signals[signal_name] = BusSignal(tensor, label, producer, timestep)

    def read(self, signal_name: str, consumer: str) -> Optional[BusSignal]:
        signal = self._signals.get(signal_name)
        if signal is not None:
            self._contract_log.append({
                "signal": signal_name,
                "producer": signal.producer,
                "label": signal.label,
                "consumer": consumer,
                "t": signal.timestep,
            })
        return signal

    def publish_handoff(self, signal_name: str, tensor: torch.Tensor,
                        producer: str, timestep: int):
        self._handoffs[signal_name] = BusSignal(
            tensor, EvidenceLabel.INFERRED, producer, timestep
        )

    def read_handoff(self, signal_name: str) -> Optional[BusSignal]:
        return self._handoffs.get(signal_name)

    def gate_cr1(self) -> Optional[torch.Tensor]:
        """CR-1: Critic A5 reroute requires Composer capability_match spread > epsilon."""
        cap = self._signals.get("capability_match")
        if cap is None:
            return None
        spread = cap.tensor.max() - cap.tensor.min()
        return (spread > self.epsilon_spread).float()

    def gate_cr2(self) -> Optional[torch.Tensor]:
        """CR-2: Permanence suppress_static_write is binding on Memory."""
        handoff = self._handoffs.get("suppress_static_write")
        if handoff is None:
            return None
        return handoff.tensor

    def gate_cr4(self, route_history: Optional[torch.Tensor] = None
                 ) -> Optional[torch.Tensor]:
        """CR-4: Tiebreak on capability ties using Critic-internal preference."""
        cap = self._signals.get("capability_match")
        if cap is None:
            return None
        scores = cap.tensor
        top2 = torch.topk(scores.flatten(), min(2, scores.numel()))
        if top2.values.numel() < 2:
            return top2.indices[0:1]
        if (top2.values[0] - top2.values[1]).abs() < self.epsilon_tie:
            if route_history is not None:
                mask = torch.ones_like(scores.flatten(), dtype=torch.bool)
                for idx in route_history.long().flatten():
                    if idx < mask.numel():
                        mask[idx] = False
                masked = scores.flatten().clone()
                masked[~mask] = -float("inf")
                return masked.argmax(dim=-1, keepdim=True)
        return top2.indices[0:1]

    def propagate_labels_cr5(self, read_labels: list) -> EvidenceLabel:
        """CR-5: Output label = MIN of all input labels."""
        if not read_labels:
            return EvidenceLabel.UNKNOWN
        return EvidenceLabel(min(read_labels))

    def get_contract_log(self) -> list:
        """CR-6: Return the audit log."""
        return self._contract_log
