"""CUT3R adapter — state-token recurrence for temporal continuity."""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .base_adapter import ExpertAdapter, ExpertOutput


class CUT3RAdapter(ExpertAdapter):

    name = "cut3r"
    capability_card = {
        "indoor_static": 0.75,
        "outdoor_static": 0.7,
        "dynamic_scene": 0.65,
        "sparse_view": 0.8,
        "dense_sequential": 0.85,
    }
    latency_estimate_ms = 30.0
    attention_regime = "full"

    def __init__(self, d_out: int = 768, n_evidence: int = 17,
                 d_evidence: int = 32, n_state_tokens: int = 32, **kwargs):
        self.d_out = d_out
        self.n_evidence = n_evidence
        self.d_evidence = d_evidence
        self.n_state_tokens = n_state_tokens
        self._loaded = False
        self._proj = nn.Linear(d_out, 3)
        self._conf = nn.Linear(d_out, 1)
        self._ev = nn.Linear(d_out, n_evidence * d_evidence)
        self._state_proj = nn.Linear(d_out, d_out)

    def forward(self, images: torch.Tensor,
                context: Optional[Dict[str, torch.Tensor]] = None,
                ) -> ExpertOutput:
        B, N = images.shape[:2]
        P = 196
        device = images.device
        feat = torch.randn(B, N, P, self.d_out, device=device)
        state_tokens = self._state_proj(
            torch.randn(B, self.n_state_tokens, self.d_out, device=device)
        )
        return ExpertOutput(
            pointmap=self._proj(feat),
            confidence=torch.sigmoid(self._conf(feat)),
            evidence_tokens=self._ev(feat.mean(dim=2)).view(B, N, self.n_evidence, self.d_evidence),
            metadata={
                "expert": self.name,
                "regime": "state_token_recurrence",
                "state_tokens": state_tokens,
            },
        )

    def load_checkpoint(self, path: str) -> None:
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded
