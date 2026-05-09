"""Fast3R adapter — speed-optimized reconstruction."""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .base_adapter import ExpertAdapter, ExpertOutput


class Fast3RAdapter(ExpertAdapter):

    name = "fast3r"
    capability_card = {
        "indoor_static": 0.7,
        "outdoor_static": 0.7,
        "dynamic_scene": 0.5,
        "sparse_view": 0.6,
        "dense_sequential": 0.8,
    }
    latency_estimate_ms = 12.0
    attention_regime = "linear"

    def __init__(self, d_out: int = 768, n_evidence: int = 17,
                 d_evidence: int = 32, **kwargs):
        self.d_out = d_out
        self.n_evidence = n_evidence
        self.d_evidence = d_evidence
        self._loaded = False
        self._proj = nn.Linear(d_out, 3)
        self._conf = nn.Linear(d_out, 1)
        self._ev = nn.Linear(d_out, n_evidence * d_evidence)

    def forward(self, images: torch.Tensor,
                context: Optional[Dict[str, torch.Tensor]] = None,
                ) -> ExpertOutput:
        B, N = images.shape[:2]
        P = 196
        device = images.device
        feat = torch.randn(B, N, P, self.d_out, device=device)
        return ExpertOutput(
            pointmap=self._proj(feat),
            confidence=torch.sigmoid(self._conf(feat)),
            evidence_tokens=self._ev(feat.mean(dim=2)).view(B, N, self.n_evidence, self.d_evidence),
            metadata={"expert": self.name, "regime": "speed_optimized"},
        )

    def load_checkpoint(self, path: str) -> None:
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded
