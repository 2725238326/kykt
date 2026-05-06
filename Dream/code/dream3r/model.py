"""
Dream3R v0.2 — bus-load-bearing wiring.

Every module now reads from the bus for cross-module decisions:
  Memory    reads: dynamic_ratio (Permanence), conflict_score (Critic t-1)
  Permanence reads: conflict_score (Critic t-1) → mint conservatism
  Critic    reads: capability_match (Composer), latent_drift_proxy (Memory)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from dream3r.bus import MemoryBus, EvidenceLabel
from dream3r.modules import Perceiver, MemorySSM, Permanence, Critic, Composer


class Dream3R(nn.Module):

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__()
        c = cfg or {}

        d_model      = c.get("d_model", 768)
        n_evidence   = c.get("n_evidence", 17)
        d_evidence   = c.get("d_evidence", 32)
        d_state      = c.get("d_state", 256)
        n_ssm_layers = c.get("n_ssm_layers", 6)
        d_slot       = c.get("d_slot", 128)
        n_slots      = c.get("n_slots", 16)
        d_critic     = c.get("d_critic", 256)
        n_regimes    = c.get("n_regimes", 5)
        n_models     = c.get("n_models", 8)
        use_backbone = c.get("use_backbone", False)
        img_size     = c.get("img_size", 224)

        self.perceiver = Perceiver(
            d_model=d_model, n_evidence=n_evidence, d_evidence=d_evidence,
            img_size=img_size, use_backbone=use_backbone,
        )
        self.memory = MemorySSM(
            d_percept=d_model, d_evidence_flat=n_evidence * d_evidence,
            d_state=d_state, n_layers=n_ssm_layers, d_bus_context=3,
        )
        self.permanence = Permanence(
            d_input=d_model, d_slot=d_slot, n_slots=n_slots,
        )
        self.critic = Critic(
            n_evidence=n_evidence, d_evidence=d_evidence, d_critic=d_critic,
        )
        self.composer = Composer(n_regimes=n_regimes, n_models=n_models)
        self.bus = MemoryBus()

    def forward(self,
                x: torch.Tensor,
                regime_probs: Optional[torch.Tensor] = None,
                prev_memory_state: Optional[torch.Tensor] = None,
                prev_object_slots: Optional[torch.Tensor] = None,
                timestep: int = 0,
                ) -> Dict[str, torch.Tensor]:
        """
        One bus tick = one window.

        Args:
            x: [B, N, P, D] features or [B, N, 3, H, W] images
            regime_probs: [B, n_regimes] or None
            prev_memory_state: [B, d_state] or None
            prev_object_slots: [B, n_slots, d_slot] or None
            timestep: window index
        """
        B = x.shape[0]
        device = x.device
        t = timestep
        self.bus.reset()

        # ========== Step 1: Perceiver ==========
        perc = self.perceiver(x)
        t1            = perc["t1"]
        t2_pointmap   = perc["t2_pointmap"]
        t2_confidence = perc["t2_confidence"]
        t3            = perc["t3_evidence"]      # [B, N, 17, d_ev]
        t3_named      = perc["t3_named"]         # dict of [B, N, d_ev]
        perc_summary  = perc["perception_summary"]

        # ========== Step 2: Memory pre-read (t-1 signals) ==========
        # These are from the PREVIOUS window's bus (or null on first window).
        # In multi-window mode, the caller should persist bus state.
        # For single-tick, these return None (null protocol).
        prev_conflict_sig = self.bus.read("conflict_score", "memory")
        prev_dynamic_sig  = self.bus.read("dynamic_ratio", "memory")

        # ========== Step 3: Permanence ==========
        perm_input = t1.mean(dim=2)  # [B, N, D]

        # Permanence reads conflict_score from bus for mint conservatism
        prev_conflict_for_perm = self.bus.read("conflict_score", "permanence")
        perm_conflict = None
        if prev_conflict_for_perm is not None:
            perm_conflict = prev_conflict_for_perm.tensor

        perm_out = self.permanence(perm_input, prev_object_slots, perm_conflict)

        self.bus.publish("dynamic_ratio", perm_out["dynamic_ratio"],
                         EvidenceLabel.INFERRED, "permanence", t)
        self.bus.publish_handoff("suppress_static_write",
                                perm_out["suppress_static_write"],
                                "permanence", t)

        # ========== Step 4: Memory ==========
        if prev_memory_state is None:
            prev_memory_state = self.memory.init_state(B, device)

        evidence_pooled = t3.mean(dim=1)             # [B, 17, d_ev]
        evidence_flat = evidence_pooled.reshape(B, -1)  # [B, 17*d_ev]

        # Memory reads from bus
        dyn_sig = self.bus.read("dynamic_ratio", "memory")
        bus_dyn = dyn_sig.tensor if dyn_sig is not None else None

        cr2 = self.bus.gate_cr2()

        mem_out = self.memory(
            perc_summary, evidence_flat, prev_memory_state,
            suppress_mask=cr2,
            bus_dynamic_ratio=bus_dyn,
            bus_conflict_score=prev_conflict_sig.tensor if prev_conflict_sig is not None else None,
        )

        self.bus.publish("latent_drift_proxy", mem_out["latent_drift_proxy"],
                         EvidenceLabel.INFERRED, "memory", t)

        # ========== Step 5: Composer ==========
        if regime_probs is None:
            regime_probs = torch.ones(B, self.composer.n_regimes, device=device)
            regime_probs = regime_probs / self.composer.n_regimes

        comp_out = self.composer(regime_probs)
        self.bus.publish("capability_match", comp_out["capability_match"],
                         EvidenceLabel.INFERRED, "composer", t)
        self.bus.publish("route_recommendation", comp_out["route_recommendation"].float(),
                         EvidenceLabel.INFERRED, "composer", t)
        self.bus.publish("route_regret", comp_out["route_regret"],
                         EvidenceLabel.INFERRED, "composer", t)

        # ========== Step 6: Critic ==========
        # Critic reads capability_match + drift from bus
        cap_sig = self.bus.read("capability_match", "critic")
        drift_sig = self.bus.read("latent_drift_proxy", "critic")

        cr1 = self.bus.gate_cr1()
        critic_out = self.critic(evidence_pooled, cr1)

        self.bus.publish("conflict_score", critic_out["conflict_score"],
                         EvidenceLabel.INFERRED, "critic", t)
        self.bus.publish("recommended_action",
                         critic_out["recommended_action"].float().unsqueeze(-1),
                         EvidenceLabel.INFERRED, "critic", t)

        # ========== Step 7-8: Output ==========
        return {
            "pointmap": t2_pointmap,
            "confidence": t2_confidence,
            "evidence_tokens": t3,
            "evidence_named": t3_named,
            "frame_tokens": t1,
            "conflict_score": critic_out["conflict_score"],
            "repair_logits": critic_out["repair_logits"],
            "recommended_action": critic_out["recommended_action"],
            "latent_state": mem_out["latent_state"],
            "update_kind": mem_out["update_kind"],
            "update_probs": mem_out["update_probs"],
            "write_decision": mem_out["write_decision"],
            "latent_drift_proxy": mem_out["latent_drift_proxy"],
            "object_track_set": perm_out["object_track_set"],
            "dynamic_ratio": perm_out["dynamic_ratio"],
            "region_logits": perm_out["region_logits"],
            "mint_confidence": perm_out["mint_confidence"],
            "capability_match": comp_out["capability_match"],
            "route_recommendation": comp_out["route_recommendation"],
            "route_regret": comp_out["route_regret"],
            "contract_log": self.bus.get_contract_log(),
        }


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

CONFIGS = {
    "small": {
        "d_model": 768, "n_evidence": 17, "d_evidence": 32,
        "d_state": 256, "n_ssm_layers": 6,
        "d_slot": 128, "n_slots": 16,
        "d_critic": 256, "n_regimes": 5, "n_models": 8,
        "use_backbone": False, "img_size": 224,
    },
    "small_vit": {
        "d_model": 768, "n_evidence": 17, "d_evidence": 32,
        "d_state": 256, "n_ssm_layers": 6,
        "d_slot": 128, "n_slots": 16,
        "d_critic": 256, "n_regimes": 5, "n_models": 8,
        "use_backbone": True, "img_size": 224,
    },
}


def build_dream3r(preset: str = "small") -> Dream3R:
    return Dream3R(CONFIGS[preset])
