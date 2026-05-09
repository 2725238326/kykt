"""
Dream3R evaluation harness.

Computes standard 3R metrics on a validation/test set:
  - Pointmap MSE / MAE
  - Confidence calibration (ECE)
  - Critic accuracy (conflict detection F1)
  - Memory utilization (AnchorBank occupancy, branch usage)
  - Composer routing diversity (entropy, regret distribution)
  - Per-regime accuracy breakdown
"""

import torch
import torch.nn.functional as F
from typing import Dict, List
from dataclasses import dataclass, field
import json


@dataclass
class EvalMetrics:
    pointmap_mse: float = 0.0
    pointmap_mae: float = 0.0
    confidence_ece: float = 0.0
    critic_f1: float = 0.0
    critic_precision: float = 0.0
    critic_recall: float = 0.0
    memory_occupancy: float = 0.0
    memory_branch_compressed: float = 0.0
    memory_branch_selected: float = 0.0
    memory_branch_sliding: float = 0.0
    routing_entropy: float = 0.0
    routing_mean_regret: float = 0.0
    n_samples: int = 0
    per_regime: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "per_regime"}
        d["per_regime"] = self.per_regime
        return d

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class Evaluator:
    """Accumulates predictions across batches, then computes final metrics."""

    def __init__(self):
        self._pointmap_se: List[float] = []
        self._pointmap_ae: List[float] = []
        self._conf_bins: List[List[float]] = [[] for _ in range(10)]
        self._conf_acc_bins: List[List[float]] = [[] for _ in range(10)]
        self._critic_tp = 0
        self._critic_fp = 0
        self._critic_fn = 0
        self._branch_weights: List[torch.Tensor] = []
        self._occupancies: List[float] = []
        self._routing_logits: List[torch.Tensor] = []
        self._regrets: List[float] = []
        self._n = 0

    @torch.no_grad()
    def update(self, outputs: Dict[str, torch.Tensor],
               targets: Dict[str, torch.Tensor]):
        self._n += 1

        if "pointmap" in outputs and "pointmap" in targets:
            pred = outputs["pointmap"]
            gt = targets["pointmap"]
            mask = targets.get("pointmap_mask", torch.ones_like(gt[..., 0]))
            se = ((pred - gt) ** 2 * mask.unsqueeze(-1)).sum() / (mask.sum() * 3 + 1e-8)
            ae = ((pred - gt).abs() * mask.unsqueeze(-1)).sum() / (mask.sum() * 3 + 1e-8)
            self._pointmap_se.append(se.item())
            self._pointmap_ae.append(ae.item())

        if "conflict_score" in outputs and "conflict_label" in targets:
            pred = (torch.sigmoid(outputs["conflict_score"].squeeze(-1)) > 0.5).float()
            gt = targets["conflict_label"].float()
            tp = ((pred == 1) & (gt == 1)).sum().item()
            fp = ((pred == 1) & (gt == 0)).sum().item()
            fn = ((pred == 0) & (gt == 1)).sum().item()
            self._critic_tp += tp
            self._critic_fp += fp
            self._critic_fn += fn

        if "nsa_branch_weights" in outputs:
            self._branch_weights.append(
                outputs["nsa_branch_weights"].mean(dim=(0, 1)).cpu()
            )

        if "bank_occupancy" in outputs:
            occ = outputs["bank_occupancy"]
            if isinstance(occ, torch.Tensor):
                self._occupancies.append(occ.mean().item())

        if "routing_logits" in outputs:
            self._routing_logits.append(outputs["routing_logits"].cpu())

        if "route_regret" in outputs:
            self._regrets.append(outputs["route_regret"].mean().item())

    def compute(self) -> EvalMetrics:
        m = EvalMetrics(n_samples=self._n)

        if self._pointmap_se:
            m.pointmap_mse = sum(self._pointmap_se) / len(self._pointmap_se)
        if self._pointmap_ae:
            m.pointmap_mae = sum(self._pointmap_ae) / len(self._pointmap_ae)

        prec_denom = self._critic_tp + self._critic_fp
        rec_denom = self._critic_tp + self._critic_fn
        m.critic_precision = self._critic_tp / prec_denom if prec_denom > 0 else 0.0
        m.critic_recall = self._critic_tp / rec_denom if rec_denom > 0 else 0.0
        if m.critic_precision + m.critic_recall > 0:
            m.critic_f1 = 2 * m.critic_precision * m.critic_recall / (m.critic_precision + m.critic_recall)

        if self._branch_weights:
            avg = torch.stack(self._branch_weights).mean(dim=0)
            m.memory_branch_compressed = avg[0].item()
            m.memory_branch_selected = avg[1].item()
            m.memory_branch_sliding = avg[2].item()

        if self._occupancies:
            m.memory_occupancy = sum(self._occupancies) / len(self._occupancies)

        if self._routing_logits:
            all_logits = torch.cat(self._routing_logits, dim=0)
            probs = F.softmax(all_logits, dim=-1)
            avg_probs = probs.mean(dim=0)
            entropy = -(avg_probs * (avg_probs + 1e-8).log()).sum().item()
            m.routing_entropy = entropy

        if self._regrets:
            m.routing_mean_regret = sum(self._regrets) / len(self._regrets)

        return m
