"""
Dream3R training losses, mapped from the training-objective sketch in SPEC-20260506-001.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class Dream3RLoss(nn.Module):
    """
    L_total = weighted sum of per-module losses.
    Each loss maps to a proxy metric from the ablation plan.
    """

    def __init__(self, weights: Dict[str, float] = None):
        super().__init__()
        self.w = weights or {
            "pointmap": 1.0,
            "critic_p1": 0.5,
            "critic_p5": 0.3,
            "memory_p2": 0.3,
            "memory_p3": 0.2,
            "permanence_p4": 0.5,
            "action_entropy": 0.1,
        }

    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses = {}
        total = torch.tensor(0.0, device=next(iter(outputs.values())).device)

        # L_pointmap: standard 3R pointmap regression
        if "pointmap" in targets:
            pred = outputs["pointmap"]
            gt = targets["pointmap"]
            mask = targets.get("pointmap_mask", torch.ones_like(gt[..., 0]))
            l = ((pred - gt) ** 2 * mask.unsqueeze(-1)).sum() / (mask.sum() * 3 + 1e-8)
            losses["pointmap"] = l
            total = total + self.w["pointmap"] * l

        # L_critic_P1: conflict detection (binary CE on conflict_score)
        if "conflict_label" in targets:
            pred = outputs["conflict_score"].squeeze(-1)
            gt = targets["conflict_label"].float()
            l = F.binary_cross_entropy_with_logits(pred, gt)
            losses["critic_p1"] = l
            total = total + self.w["critic_p1"] * l

        # L_critic_P5: repair action (CE on repair_logits)
        if "repair_label" in targets:
            pred = outputs["repair_logits"]
            pred = pred.clamp(min=-60000, max=60000)
            gt = targets["repair_label"].long()
            l = F.cross_entropy(pred, gt)
            losses["critic_p5"] = l
            total = total + self.w["critic_p5"] * l

        # L_memory_P2: anchor retention (encourage high scores on important anchors)
        if "anchor_importance_label" in targets:
            pred = outputs.get("update_kind")
            if pred is not None:
                gt = targets["anchor_importance_label"]
                l = F.cross_entropy(pred, gt.long())
                losses["memory_p2"] = l
                total = total + self.w["memory_p2"] * l

        # L_permanence_P4: dynamic pollution (CE on region classifier)
        if "region_label" in targets:
            pred = outputs["region_logits"]
            gt = targets["region_label"].long()
            B, S, C = pred.shape
            l = F.cross_entropy(pred.view(B * S, C), gt.view(B * S))
            losses["permanence_p4"] = l
            total = total + self.w["permanence_p4"] * l

        # L_action_entropy: encourage non-degenerate action distributions
        for key in ["update_kind", "repair_logits", "region_logits"]:
            if key in outputs:
                logits = outputs[key]
                if logits.dim() == 3:
                    logits = logits.reshape(-1, logits.shape[-1])
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
                losses[f"entropy_{key}"] = entropy
                total = total + self.w["action_entropy"] * (-entropy)

        losses["total"] = total
        return losses
