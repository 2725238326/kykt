"""
Dream3R expert adapter registry and factory.

Usage:
    from dream3r.composer_experts import ExpertRegistry, get_all_adapters
    registry = ExpertRegistry()
    registry.register_all_defaults()
    adapter = registry.get("mast3r")
"""

from typing import Dict, List, Optional, Type
import torch

from .base_adapter import ExpertAdapter, ExpertOutput


class ExpertRegistry:
    """Central registry for expert adapters."""

    def __init__(self):
        self._adapters: Dict[str, ExpertAdapter] = {}
        self._classes: Dict[str, Type[ExpertAdapter]] = {}

    def register_class(self, name: str, cls: Type[ExpertAdapter]):
        self._classes[name] = cls

    def instantiate(self, name: str, **kwargs) -> ExpertAdapter:
        if name not in self._classes:
            raise KeyError(f"Unknown expert: {name}. Available: {list(self._classes.keys())}")
        adapter = self._classes[name](**kwargs)
        self._adapters[name] = adapter
        return adapter

    def get(self, name: str) -> ExpertAdapter:
        if name not in self._adapters:
            return self.instantiate(name)
        return self._adapters[name]

    def register_all_defaults(self):
        from .mast3r_adapter import MASt3RAdapter
        from .fast3r_adapter import Fast3RAdapter
        from .spann3r_adapter import Spann3RAdapter
        from .cut3r_adapter import CUT3RAdapter
        from .moge2_adapter import MoGe2Adapter
        from .depthanything_adapter import DepthAnythingAdapter
        from .test3r_adapter import Test3RAdapter

        for cls in [MASt3RAdapter, Fast3RAdapter, Spann3RAdapter,
                    CUT3RAdapter, MoGe2Adapter, DepthAnythingAdapter,
                    Test3RAdapter]:
            self.register_class(cls.name, cls)

    @property
    def names(self) -> List[str]:
        return list(self._classes.keys())

    def capability_matrix(self, regime_order: Optional[list] = None) -> torch.Tensor:
        order = regime_order or ExpertAdapter.REGIMES
        rows = []
        for name in sorted(self._classes.keys()):
            adapter = self.get(name)
            rows.append(adapter.capability_tensor(order))
        return torch.stack(rows)

    def latency_vector(self) -> torch.Tensor:
        return torch.tensor([
            self.get(n).latency_estimate_ms
            for n in sorted(self._classes.keys())
        ])


def get_all_adapters() -> ExpertRegistry:
    reg = ExpertRegistry()
    reg.register_all_defaults()
    return reg
