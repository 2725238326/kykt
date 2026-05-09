"""
Dream3R v0.3 — full bus-load-bearing orchestrator with NSA Memory + Composer Router.

Supports both v0.1 (MemorySSM + table Composer) and v0.3 (SpatialMemory +
ComposerRouter) configurations via the `version` config key.

Every module reads from the bus for cross-module decisions:
  Memory      reads: dynamic_ratio (Permanence), conflict_score (Critic t-1)
  Permanence  reads: conflict_score (Critic t-1) -> mint conservatism
  Critic      reads: capability_match (Composer), latent_drift_proxy (Memory)
  Composer    reads: critic confidence for routing decisions (v0.3 only)
"""

import time
import torch
import torch.nn as nn
from typing import Dict, List, Optional

from dream3r.bus import MemoryBus, EvidenceLabel
from dream3r.modules import (
    Perceiver, Permanence, Critic,
    MemorySSM_v01, Composer_v01,
    SpatialMemory, ComposerRouter,
)
from dream3r.composer_experts import ExpertRegistry


class Dream3R(nn.Module):

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__()
        c = cfg or {}
        version = c.get("version", "v03")

        d_model      = c.get("d_model", 768)
        n_evidence   = c.get("n_evidence", 17)
        d_evidence   = c.get("d_evidence", 32)
        use_backbone = c.get("use_backbone", False)
        img_size     = c.get("img_size", 224)

        self.version = version
        self.profile = c.get("profile", False)
        self._timings: Dict[str, List[float]] = {}

        self.perceiver = Perceiver(
            d_model=d_model, n_evidence=n_evidence, d_evidence=d_evidence,
            img_size=img_size, use_backbone=use_backbone,
        )

        if version == "v01":
            d_state      = c.get("d_state", 256)
            n_ssm_layers = c.get("n_ssm_layers", 6)
            d_slot       = c.get("d_slot", 128)
            n_slots      = c.get("n_slots", 16)
            d_critic     = c.get("d_critic", 256)
            n_regimes    = c.get("n_regimes", 5)
            n_models     = c.get("n_models", 8)

            self.memory = MemorySSM_v01(
                d_percept=d_model, d_evidence_flat=n_evidence * d_evidence,
                d_state=d_state, n_layers=n_ssm_layers, d_bus_context=3,
            )
            self.permanence = Permanence(
                d_input=d_model, d_slot=d_slot, n_slots=n_slots,
            )
            self.critic = Critic(
                n_evidence=n_evidence, d_evidence=d_evidence, d_critic=d_critic,
            )
            self.composer = Composer_v01(n_regimes=n_regimes, n_models=n_models)
        else:
            d_mem        = c.get("d_memory", 128)
            n_state      = c.get("n_state_tokens", 32)
            bank_cap     = c.get("bank_capacity", 256)
            nsa_k        = c.get("nsa_select_k", 8)
            nsa_heads    = c.get("nsa_heads", 4)
            slide_win    = c.get("sliding_window", 4)
            d_slot       = c.get("d_slot", 128)
            n_slots      = c.get("n_slots", 16)
            d_critic     = c.get("d_critic", 256)
            n_regimes    = c.get("n_regimes", 5)
            d_routing    = c.get("d_routing", 64)
            cost_alpha   = c.get("cost_alpha", 0.5)

            registry = ExpertRegistry()
            registry.register_all_defaults()

            self.memory = SpatialMemory(
                d_model=d_mem, n_state_tokens=n_state,
                bank_capacity=bank_cap, nsa_n_select_k=nsa_k,
                nsa_n_heads=nsa_heads, sliding_window=slide_win,
                n_evidence=n_evidence, d_evidence=d_evidence,
            )
            self.permanence = Permanence(
                d_input=d_model, d_slot=d_slot, n_slots=n_slots,
            )
            self.critic = Critic(
                n_evidence=n_evidence, d_evidence=d_evidence, d_critic=d_critic,
            )
            self.composer = ComposerRouter(
                n_regimes=n_regimes, d_routing=d_routing,
                cost_alpha=cost_alpha, expert_registry=registry,
            )
            self.composer.load_from_registry()

        self.bus = MemoryBus()

    def _time(self, name: str):
        """Context-manager-free timing. Call _time_start/_time_end."""
        pass

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
            prev_memory_state: [B, d_state] (v01) or [B, S, D] (v03) or None
            prev_object_slots: [B, n_slots, d_slot] or None
            timestep: window index
        """
        B = x.shape[0]
        device = x.device
        t = timestep
        self.bus.reset()
        timings = {}

        t0 = time.perf_counter()
        perc = self.perceiver(x)
        if self.profile:
            timings["perceiver_ms"] = (time.perf_counter() - t0) * 1000

        t1            = perc["t1"]
        t2_pointmap   = perc["t2_pointmap"]
        t2_confidence = perc["t2_confidence"]
        t3            = perc["t3_evidence"]
        t3_named      = perc["t3_named"]
        perc_summary  = perc["perception_summary"]

        prev_conflict_sig = self.bus.read("conflict_score", "memory")
        prev_dynamic_sig  = self.bus.read("dynamic_ratio", "memory")

        # === Permanence ===
        t0 = time.perf_counter()
        perm_input = t1.mean(dim=2)
        prev_conflict_for_perm = self.bus.read("conflict_score", "permanence")
        perm_conflict = None
        if prev_conflict_for_perm is not None:
            perm_conflict = prev_conflict_for_perm.tensor

        perm_out = self.permanence(perm_input, prev_object_slots, perm_conflict)
        if self.profile:
            timings["permanence_ms"] = (time.perf_counter() - t0) * 1000

        self.bus.publish("dynamic_ratio", perm_out["dynamic_ratio"],
                         EvidenceLabel.INFERRED, "permanence", t)
        self.bus.publish_handoff("suppress_static_write",
                                perm_out["suppress_static_write"],
                                "permanence", t)

        # === Memory ===
        t0 = time.perf_counter()
        dyn_sig = self.bus.read("dynamic_ratio", "memory")
        bus_dyn = dyn_sig.tensor if dyn_sig is not None else None
        cr2 = self.bus.gate_cr2()

        evidence_pooled = t3.mean(dim=1)
        evidence_flat = evidence_pooled.reshape(B, -1)

        if self.version == "v01":
            if prev_memory_state is None:
                prev_memory_state = self.memory.init_state(B, device)
            mem_out = self.memory(
                perc_summary, evidence_flat, prev_memory_state,
                suppress_mask=cr2,
                bus_dynamic_ratio=bus_dyn,
                bus_conflict_score=prev_conflict_sig.tensor if prev_conflict_sig is not None else None,
            )
        else:
            frame_tokens_flat = t1.mean(dim=1)
            if prev_memory_state is None:
                prev_memory_state = self.memory.init_state(B, device)
            mem_out = self.memory(
                frame_tokens_flat, evidence_flat, prev_memory_state,
                bus_dynamic_ratio=bus_dyn,
                bus_conflict_score=prev_conflict_sig.tensor if prev_conflict_sig is not None else None,
                suppress_mask=cr2,
            )

        if self.profile:
            timings["memory_ms"] = (time.perf_counter() - t0) * 1000

        self.bus.publish("latent_drift_proxy", mem_out["latent_drift_proxy"],
                         EvidenceLabel.INFERRED, "memory", t)

        if "bank_occupancy" in mem_out:
            self.bus.publish("bank_occupancy", mem_out["bank_occupancy"],
                             EvidenceLabel.INFERRED, "memory", t)

        # === Composer ===
        t0 = time.perf_counter()
        n_regimes = 5
        if isinstance(self.composer, ComposerRouter):
            n_regimes = self.composer.n_regimes
        elif isinstance(self.composer, Composer_v01):
            n_regimes = self.composer.n_regimes

        if regime_probs is None:
            regime_probs = torch.ones(B, n_regimes, device=device) / n_regimes

        if self.version == "v01":
            comp_out = self.composer(regime_probs)
        else:
            critic_conf = None
            cs = self.bus.read("conflict_score", "composer")
            if cs is not None:
                critic_conf = 1.0 - torch.sigmoid(cs.tensor)
            comp_out = self.composer(regime_probs, critic_confidence=critic_conf)

        if self.profile:
            timings["composer_ms"] = (time.perf_counter() - t0) * 1000

        self.bus.publish("capability_match", comp_out["capability_match"],
                         EvidenceLabel.INFERRED, "composer", t)
        self.bus.publish("route_recommendation", comp_out["route_recommendation"].float(),
                         EvidenceLabel.INFERRED, "composer", t)
        self.bus.publish("route_regret", comp_out["route_regret"],
                         EvidenceLabel.INFERRED, "composer", t)

        if "selected_expert" in comp_out:
            self.bus.publish("routed_expert_id", comp_out["selected_expert"].float().unsqueeze(-1),
                             EvidenceLabel.INFERRED, "composer", t)

        # === Critic ===
        t0 = time.perf_counter()
        cap_sig = self.bus.read("capability_match", "critic")
        drift_sig = self.bus.read("latent_drift_proxy", "critic")

        cr1 = self.bus.gate_cr1()
        critic_out = self.critic(evidence_pooled, cr1)

        if self.profile:
            timings["critic_ms"] = (time.perf_counter() - t0) * 1000

        self.bus.publish("conflict_score", critic_out["conflict_score"],
                         EvidenceLabel.INFERRED, "critic", t)
        self.bus.publish("recommended_action",
                         critic_out["recommended_action"].float().unsqueeze(-1),
                         EvidenceLabel.INFERRED, "critic", t)

        # === Output ===
        result = {
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

        if "nsa_branch_weights" in mem_out:
            result["nsa_branch_weights"] = mem_out["nsa_branch_weights"]
            result["bank_occupancy"] = mem_out["bank_occupancy"]
        if "latent_state_tokens" in mem_out:
            result["latent_state_tokens"] = mem_out["latent_state_tokens"]
        if "selected_expert" in comp_out:
            result["selected_expert"] = comp_out["selected_expert"]
            result["routing_logits"] = comp_out["routing_logits"]
            result["cost_adjusted_scores"] = comp_out["cost_adjusted_scores"]

        if self.profile:
            result["timings"] = timings

        return result


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

CONFIGS = {
    "small_v01": {
        "version": "v01",
        "d_model": 768, "n_evidence": 17, "d_evidence": 32,
        "d_state": 256, "n_ssm_layers": 6,
        "d_slot": 128, "n_slots": 16,
        "d_critic": 256, "n_regimes": 5, "n_models": 8,
        "use_backbone": False, "img_size": 224,
    },
    "small": {
        "version": "v03",
        "d_model": 768, "n_evidence": 17, "d_evidence": 32,
        "d_memory": 128, "n_state_tokens": 32,
        "bank_capacity": 256, "nsa_select_k": 8, "nsa_heads": 4,
        "sliding_window": 4,
        "d_slot": 128, "n_slots": 16,
        "d_critic": 256, "n_regimes": 5,
        "d_routing": 64, "cost_alpha": 0.5,
        "use_backbone": False, "img_size": 224,
    },
    "small_vit": {
        "version": "v03",
        "d_model": 768, "n_evidence": 17, "d_evidence": 32,
        "d_memory": 128, "n_state_tokens": 32,
        "bank_capacity": 256, "nsa_select_k": 8, "nsa_heads": 4,
        "sliding_window": 4,
        "d_slot": 128, "n_slots": 16,
        "d_critic": 256, "n_regimes": 5,
        "d_routing": 64, "cost_alpha": 0.5,
        "use_backbone": True, "img_size": 224,
    },
    "base": {
        "version": "v03",
        "d_model": 768, "n_evidence": 17, "d_evidence": 32,
        "d_memory": 256, "n_state_tokens": 64,
        "bank_capacity": 512, "nsa_select_k": 16, "nsa_heads": 8,
        "sliding_window": 8,
        "d_slot": 256, "n_slots": 32,
        "d_critic": 512, "n_regimes": 5,
        "d_routing": 128, "cost_alpha": 0.5,
        "use_backbone": True, "img_size": 224,
    },
}


def build_dream3r(preset: str = "small") -> Dream3R:
    return Dream3R(CONFIGS[preset])
