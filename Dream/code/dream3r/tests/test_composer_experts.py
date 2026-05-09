"""Unit tests for composer expert adapters."""

import torch
from dream3r.composer_experts import ExpertRegistry
from dream3r.composer_experts.base_adapter import ExpertAdapter


def test_registry_registration():
    reg = ExpertRegistry()
    reg.register_all_defaults()
    assert len(reg.names) == 7
    expected = {"mast3r", "fast3r", "spann3r", "cut3r", "moge2", "depthanything", "test3r"}
    assert set(reg.names) == expected


def test_capability_matrix():
    reg = ExpertRegistry()
    reg.register_all_defaults()
    matrix = reg.capability_matrix()
    assert matrix.shape == (7, 5)
    assert (matrix >= 0).all() and (matrix <= 1).all()


def test_latency_vector():
    reg = ExpertRegistry()
    reg.register_all_defaults()
    latency = reg.latency_vector()
    assert latency.shape == (7,)
    assert (latency > 0).all()

    fastest = reg.get("depthanything")
    assert fastest.latency_estimate_ms < 20

    slowest = reg.get("test3r")
    assert slowest.latency_estimate_ms > 100


def test_adapter_forward():
    reg = ExpertRegistry()
    reg.register_all_defaults()

    for name in reg.names:
        adapter = reg.get(name)
        images = torch.randn(2, 4, 3, 224, 224)
        out = adapter.forward(images)
        assert out.pointmap.shape == (2, 4, 196, 3), f"{name} pointmap shape wrong"
        assert out.confidence.shape == (2, 4, 196, 1), f"{name} confidence shape wrong"
        assert out.evidence_tokens.shape == (2, 4, 17, 32), f"{name} evidence shape wrong"
        assert "expert" in out.metadata


def test_capability_tensor():
    reg = ExpertRegistry()
    reg.register_all_defaults()

    adapter = reg.get("mast3r")
    ct = adapter.capability_tensor()
    assert ct.shape == (5,)
    assert ct.sum() > 0


def test_cut3r_state_tokens():
    reg = ExpertRegistry()
    reg.register_all_defaults()
    adapter = reg.get("cut3r")
    out = adapter.forward(torch.randn(1, 2, 3, 224, 224))
    assert "state_tokens" in out.metadata
    assert out.metadata["state_tokens"].shape[1] == 32


if __name__ == "__main__":
    test_registry_registration()
    test_capability_matrix()
    test_latency_vector()
    test_adapter_forward()
    test_capability_tensor()
    test_cut3r_state_tokens()
    print("All composer expert tests passed.")
