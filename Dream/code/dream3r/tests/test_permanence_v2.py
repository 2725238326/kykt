"""Tests for per-slot Permanence and CR-2 aggregation."""

import torch

from dream3r.bus import EvidenceLabel, MemoryBus
from dream3r.modules import Permanence


def test_permanence_returns_per_slot_dynamic_ratio_and_suppress():
    torch.manual_seed(0)
    perm = Permanence(d_input=32, d_slot=16, n_slots=4, n_iters=1)
    out = perm(torch.randn(2, 5, 32))

    assert out["dynamic_ratio"].shape == (2, 4, 1)
    assert out["suppress_static_write"].shape == (2, 4)
    assert out["slot_match_indices"].shape == (2, 4)


def test_cr2_aggregation_is_not_any_slot_suppress():
    bus = MemoryBus()
    per_slot = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    bus.publish_handoff("suppress_static_write", per_slot, "permanence", timestep=0)

    assert torch.equal(bus.cr2_per_slot_suppress(), per_slot)
    assert bus.gate_cr2().item() == 0.0

    bus.publish_handoff("suppress_static_write", torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
                        "permanence", timestep=1)
    assert bus.gate_cr2().item() == 1.0


def test_slot_matching_tracks_previous_slot_identity():
    torch.manual_seed(0)
    perm = Permanence(d_input=32, d_slot=16, n_slots=4, n_iters=1)
    first = perm(torch.randn(1, 6, 32))
    second = perm(torch.randn(1, 6, 32), prev_slots=first["object_track_set"])

    assert second["slot_match_indices"].shape == (1, 4)
    assert second["slot_match_indices"].min() >= 0
    assert second["slot_match_indices"].max() < 4
    assert torch.isfinite(second["slot_match_scores"]).all()


def test_cr3_permanence_bias_aggregates_per_slot_dynamic_ratio():
    bus = MemoryBus()
    dyn = torch.tensor([[[0.2], [0.4], [0.6], [0.8]]])
    bus.publish("dynamic_ratio", dyn, EvidenceLabel.INFERRED, "permanence", timestep=0)

    bias = bus.cr3_permanence_bias()
    assert bias.shape == (1, 1)
    assert torch.allclose(bias, torch.tensor([[0.5]]))


if __name__ == "__main__":
    test_permanence_returns_per_slot_dynamic_ratio_and_suppress()
    test_cr2_aggregation_is_not_any_slot_suppress()
    test_slot_matching_tracks_previous_slot_identity()
    test_cr3_permanence_bias_aggregates_per_slot_dynamic_ratio()
    print("All Permanence v2 tests passed.")
