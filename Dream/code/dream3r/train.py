"""
Dream3R training script.

Supports:
  - DDP multi-GPU (2-3 cards, CUDA_VISIBLE_DEVICES to pick)
  - Mixed precision (torch.amp)
  - Checkpoint save/resume
  - TensorBoard logging
  - Gradient clipping

Usage:
    # Single GPU
    python -m dream3r.train --preset small --gpus 0

    # 2-GPU DDP
    torchrun --nproc_per_node=2 -m dream3r.train --preset small --gpus 0,1

    # Resume from checkpoint
    torchrun --nproc_per_node=2 -m dream3r.train --resume checkpoints/latest.pt
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from dream3r.config import load_config, save_config, config_to_model_args
from dream3r.model import Dream3R
from dream3r.losses import Dream3RLoss


# ---------------------------------------------------------------------------
# Placeholder dataset (replace with DTU/KITTI loader)
# ---------------------------------------------------------------------------

class SyntheticDataset(Dataset):
    """Synthetic data for testing the training loop before real data is ready."""

    def __init__(self, n_samples: int = 200, n_frames: int = 4,
                 n_patches: int = 196, d_model: int = 768):
        self.n = n_samples
        self.n_frames = n_frames
        self.n_patches = n_patches
        self.d_model = d_model

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = torch.randn(self.n_frames, self.n_patches, self.d_model)
        targets = {
            "pointmap": torch.randn(self.n_frames, self.n_patches, 3),
            "pointmap_mask": torch.ones(self.n_frames, self.n_patches),
            "conflict_label": torch.randint(0, 2, ()).float(),
            "repair_label": torch.randint(0, 6, ()),
            "region_label": torch.randint(0, 3, (16,)),
        }
        return x, targets


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def setup_ddp(rank: int, world_size: int, backend: str = "nccl"):
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def save_checkpoint(model: nn.Module, optimizer, scaler, epoch: int,
                    step: int, cfg: dict, path: str):
    state = {
        "epoch": epoch,
        "step": step,
        "cfg": cfg,
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m = model.module if hasattr(model, "module") else model
    m.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("epoch", 0), ckpt.get("step", 0)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: dict):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp = world_size > 1

    if is_ddp:
        setup_ddp(local_rank, world_size, cfg["dist_backend"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Model
    model_cfg = config_to_model_args(cfg)
    model = Dream3R(model_cfg).to(device)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Dream3R: {n_params:,} params | {world_size} GPU(s) | AMP={cfg['amp']}")

    # Loss
    loss_fn = Dream3RLoss(weights={
        "pointmap": cfg["w_pointmap"],
        "critic_p1": cfg["w_critic_p1"],
        "critic_p5": cfg["w_critic_p5"],
        "memory_p2": cfg["w_memory_p2"],
        "memory_p3": cfg["w_memory_p3"],
        "permanence_p4": cfg["w_permanence_p4"],
        "action_entropy": cfg["w_action_entropy"],
    }).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda") if cfg["amp"] else None

    # Dataset
    if cfg["dataset"] == "synthetic":
        dataset = SyntheticDataset(
            n_samples=200, n_frames=cfg["n_frames_per_window"],
            n_patches=196, d_model=cfg["d_model"],
        )
        sampler = DistributedSampler(dataset) if is_ddp else None
        loader = DataLoader(
            dataset, batch_size=cfg["batch_size"],
            sampler=sampler, shuffle=(sampler is None),
            num_workers=cfg["num_workers"], pin_memory=True,
            drop_last=True,
        )
    elif cfg["dataset"] == "dtu":
        from dream3r.data_dtu import build_dtu_loaders
        loader, val_loader, sampler = build_dtu_loaders(cfg)
        if is_main_process():
            print(f"DTU: {len(loader.dataset)} train, {len(val_loader.dataset)} val samples")
    else:
        raise ValueError(f"Unknown dataset: {cfg['dataset']}")

    # TensorBoard
    writer = None
    if is_main_process():
        try:
            from tensorboardX import SummaryWriter
            log_dir = Path(cfg["log_dir"]) / time.strftime("%Y%m%d-%H%M%S")
            writer = SummaryWriter(str(log_dir))
            save_config(cfg, str(log_dir / "config.yaml"))
            print(f"Logging to {log_dir}")
        except ImportError:
            print("tensorboardX not found, skipping logging")

    # Resume
    start_epoch, global_step = 0, 0
    resume_path = cfg.get("resume")
    if resume_path and Path(resume_path).exists():
        start_epoch, global_step = load_checkpoint(resume_path, model, optimizer, scaler)
        if is_main_process():
            print(f"Resumed from {resume_path} (epoch {start_epoch}, step {global_step})")

    # Train
    for epoch in range(start_epoch, cfg["epochs"]):
        if sampler:
            sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (x, targets) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

            optimizer.zero_grad(set_to_none=True)

            if cfg["amp"]:
                with torch.amp.autocast("cuda"):
                    outputs = model(x)
                    losses = loss_fn(outputs, targets)
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(x)
                losses = loss_fn(outputs, targets)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()

            epoch_loss += losses["total"].item()
            global_step += 1

            if is_main_process() and global_step % cfg["log_every"] == 0:
                if writer:
                    for k, v in losses.items():
                        writer.add_scalar(f"loss/{k}", v.item(), global_step)
                    if "update_probs" in outputs:
                        probs = outputs["update_probs"].detach().mean(0)
                        for i, name in enumerate(["full", "pose_adpt", "kalman", "skip", "reset"]):
                            writer.add_scalar(f"a1_mode/{name}", probs[i].item(), global_step)

        dt = time.time() - t0
        avg = epoch_loss / max(len(loader), 1)

        if is_main_process():
            print(f"Epoch {epoch+1}/{cfg['epochs']}  loss={avg:.4f}  time={dt:.1f}s")
            if writer:
                writer.add_scalar("epoch/loss", avg, epoch)

            if (epoch + 1) % cfg["save_every_epoch"] == 0:
                path = Path(cfg["save_dir"]) / f"epoch_{epoch+1:04d}.pt"
                save_checkpoint(model, optimizer, scaler, epoch + 1, global_step, cfg, str(path))
                print(f"  Saved {path}")

    # Final save
    if is_main_process():
        path = Path(cfg["save_dir"]) / "latest.pt"
        save_checkpoint(model, optimizer, scaler, cfg["epochs"], global_step, cfg, str(path))
        print(f"Training done. Final checkpoint: {path}")
        if writer:
            writer.close()

    cleanup_ddp()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dream3R training")
    parser.add_argument("--preset", default="small", help="Config preset name")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--resume", default=None, help="Path to checkpoint")
    parser.add_argument("--gpus", default=None, help="GPU ids (e.g. 0,1)")
    parser.add_argument("--dataset", default=None, help="Dataset: synthetic/dtu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.resume:
        overrides["resume"] = args.resume
    if args.gpus:
        overrides["gpus"] = args.gpus
    if args.dataset:
        overrides["dataset"] = args.dataset
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    if args.lr:
        overrides["lr"] = args.lr

    cfg = load_config(path=args.config, preset=args.preset, overrides=overrides)

    if "WORLD_SIZE" not in os.environ:
        gpu_list = cfg["gpus"].split(",")
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg["gpus"]
        if len(gpu_list) == 1:
            train(cfg)
        else:
            print(f"Use torchrun for multi-GPU: torchrun --nproc_per_node={len(gpu_list)} -m dream3r.train")
            sys.exit(1)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
