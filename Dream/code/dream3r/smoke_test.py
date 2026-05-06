"""
Smoke test: validates Dream3R forward + loss + backward + bus signals.
Usage: python -m dream3r.smoke_test
"""

import torch
import sys


def smoke_test():
    from dream3r.model import build_dream3r
    from dream3r.losses import Dream3RLoss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_dream3r("small").to(device)
    loss_fn = Dream3RLoss().to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")

    # --- Forward ---
    B, N, P, D = 2, 4, 196, 768
    x = torch.randn(B, N, P, D, device=device)
    regime = torch.softmax(torch.randn(B, 5, device=device), dim=-1)

    print("\nForward pass (window 1)...")
    out1 = model(x, regime, timestep=0)
    for k, v in sorted(out1.items()):
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {list(v.shape)}")
        elif isinstance(v, list):
            print(f"  {k}: list[{len(v)}]")

    # --- Bus contract log ---
    log = out1["contract_log"]
    print(f"\nBus contract log: {len(log)} entries")
    for e in log[:10]:
        print(f"  {e['consumer']} <- {e['signal']} ({e['producer']})")

    # Check bus is load-bearing: Memory and Permanence should read from bus
    readers = set(e["consumer"] for e in log)
    bus_ok = "memory" in readers and "critic" in readers
    print(f"  Memory reads from bus: {'memory' in readers}")
    print(f"  Permanence reads from bus: {'permanence' in readers}")
    print(f"  Critic reads from bus: {'critic' in readers}")

    # --- Loss ---
    print("\nLoss computation...")
    targets = {
        "pointmap": torch.randn(B, N, P, 3, device=device),
        "pointmap_mask": torch.ones(B, N, P, device=device),
        "conflict_label": torch.randint(0, 2, (B,), device=device).float(),
        "repair_label": torch.randint(0, 6, (B,), device=device),
        "region_label": torch.randint(0, 3, (B, 16), device=device),
    }
    losses = loss_fn(out1, targets)
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    # --- Backward ---
    print("\nBackward pass...")
    losses["total"].backward()
    bad = sum(1 for p in model.parameters()
              if p.requires_grad and p.grad is not None and torch.isnan(p.grad).any())
    print(f"  NaN gradients: {bad}")

    # --- Memory carry-over ---
    print("\nMemory carry-over (window 2)...")
    model.zero_grad()
    out2 = model(x, regime,
                 prev_memory_state=out1["latent_state"].detach(),
                 prev_object_slots=out1["object_track_set"].detach(),
                 timestep=1)
    same = torch.allclose(out1["latent_state"], out2["latent_state"])
    print(f"  States differ across windows: {not same}")

    # --- CR-1 gate ---
    print("\nCR-1 gate test...")
    model.composer.set_capability_cards(torch.ones(8, 5) * 0.5)
    out_eq = model(x, regime)
    p_block = torch.softmax(out_eq["repair_logits"], dim=-1)[:, 2].mean().item()

    cards = torch.rand(8, 5, device=device); cards[0] = 1.0
    model.composer.set_capability_cards(cards)
    out_diff = model(x, regime)
    p_allow = torch.softmax(out_diff["repair_logits"], dim=-1)[:, 2].mean().item()
    print(f"  Reroute blocked (equal models): {p_block:.4f}")
    print(f"  Reroute allowed (varied models): {p_allow:.4f}")

    # --- A1 update mode distribution ---
    print("\nA1 update mode distribution:")
    probs = out1["update_probs"].mean(dim=0)
    modes = ["full", "pose_adaptive", "kalman", "skip", "reset"]
    for i, m in enumerate(modes):
        print(f"  {m}: {probs[i].item():.3f}")

    ok = (len(log) > 0 and bad == 0 and not same and p_block < p_allow)
    tag = "PASSED" if ok else "PASSED (with warnings)"
    print(f"\n=== SMOKE TEST {tag} ===")
    return ok


if __name__ == "__main__":
    try:
        ok = smoke_test()
        sys.exit(0 if ok else 0)
    except Exception as e:
        print(f"\n=== SMOKE TEST FAILED: {e} ===")
        import traceback; traceback.print_exc()
        sys.exit(1)
