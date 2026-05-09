"""Spann3R adapter — sequential/streaming spatial reconstruction."""

import torch
from typing import Dict, Optional

from .base_adapter import ExpertAdapter, ExpertOutput
from .fallback import image_fallback_output


class Spann3RAdapter(ExpertAdapter):

    name = "spann3r"
    capability_card = {
        "indoor_static": 0.6,
        "outdoor_static": 0.7,
        "dynamic_scene": 0.4,
        "sparse_view": 0.5,
        "dense_sequential": 0.95,
    }
    latency_estimate_ms = 28.0
    attention_regime = "sparse"

    def __init__(self, d_out: int = 768, n_evidence: int = 17,
                 d_evidence: int = 32, **kwargs):
        self.d_out = d_out
        self.n_evidence = n_evidence
        self.d_evidence = d_evidence
        self._loaded = False

    def forward(self, images: torch.Tensor,
                context: Optional[Dict[str, torch.Tensor]] = None,
                ) -> ExpertOutput:
        return image_fallback_output(
            images, self.name, "streaming_spatial",
            self.n_evidence, self.d_evidence,
            metadata={"attention_regime": self.attention_regime},
        )

    def load_checkpoint(self, path: str) -> None:
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded
