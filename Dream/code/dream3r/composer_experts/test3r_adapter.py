"""Test3R adapter — offline verification expert (heavyweight, lazy-invoked)."""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .base_adapter import ExpertAdapter, ExpertOutput


class Test3RAdapter(ExpertAdapter):
    """
    Test3R is the heavyweight verification expert. It runs OFF the streaming
    path — only invoked when the Critic triggers test-time verification.
    Its high latency makes it unsuitable for real-time routing.
    """

    name = "test3r"
    capability_card = {
        "indoor_static": 0.95,
        "outdoor_static": 0.9,
        "dynamic_scene": 0.4,
        "sparse_view": 0.85,
        "dense_sequential": 0.3,
    }
    latency_estimate_ms = 120.0
    attention_regime = "full"

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
            metadata={"expert": self.name, "regime": "offline_verification"},
        )

    def load_checkpoint(self, path: str) -> None:
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded
