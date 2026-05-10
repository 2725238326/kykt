from pathlib import Path
p = Path('/hdd3/kykt26/code/dream3r/dream3r/modules.py')
text = p.read_text()
start = text.index('class Perceiver(nn.Module):')
end = text.index('\n\n# ---------------------------------------------------------------------------\n# C2:', start)
new = r'''class Perceiver(nn.Module):
    """Per-frame backbone plus trainable geometry/evidence heads."""

    EVIDENCE_SIGNALS = [
        "pose_novelty", "view_overlap", "reprojection_residual",
        "pointmap_conflict", "confidence_drop", "latent_drift_proxy",
        "dynamic_ratio", "optical_flow_conflict", "object_track_stability",
        "loop_candidate_score", "anchor_importance", "cache_pressure",
        "external_memory_overlap", "prior_rgb_conflict",
        "blur_or_low_light_score", "uncertainty_area",
        "model_capability_match",
    ]

    DINO_HUB_NAMES = {
        "dinov2_vitb14": "dinov2_vitb14",
        "dinov2_vitl14": "dinov2_vitl14",
        "dinov3": "dinov2_vitb14",
    }

    def __init__(self, d_model: int = 768, n_evidence: int = 17,
                 d_evidence: int = 32, img_size: int = 224,
                 patch_size: int = 16, use_backbone: bool = True,
                 backbone_type: str = "none",
                 backbone_freeze: bool = True,
                 backbone_checkpoint_path: str = ""):
        super().__init__()
        self.d_model = d_model
        self.n_evidence = n_evidence
        self.d_evidence = d_evidence
        self.use_backbone = use_backbone
        self.backbone_type = backbone_type or "none"
        self.backbone_freeze = backbone_freeze
        self.backbone_checkpoint_path = backbone_checkpoint_path or ""
        self.backbone_load_error = None
        self.backbone = None
        self.backbone_dim = d_model
        self.backbone_proj = nn.Identity()

        if use_backbone and self.backbone_type not in {"none", "identity"}:
            self._try_load_backbone()
        elif use_backbone and self.backbone_type in {"none", "identity"}:
            self._try_load_timm_backbone(pretrained=False)

        self.pointmap_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
        self.evidence_projectors = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(d_model, d_model // 4), nn.GELU(),
                nn.Linear(d_model // 4, d_evidence),
            )
            for name in self.EVIDENCE_SIGNALS
        })

    def _finalize_backbone(self, backbone: nn.Module, backbone_dim: int):
        self.backbone = backbone
        self.backbone_dim = backbone_dim
        if self.backbone_freeze:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        self.backbone_proj = (
            nn.Linear(backbone_dim, self.d_model)
            if backbone_dim != self.d_model else nn.Identity()
        )
        self.backbone_load_error = None

    def _try_load_backbone(self):
        try:
            if self.backbone_type.startswith("dinov2") or self.backbone_type == "dinov3":
                self._load_dino_backbone()
            else:
                self._try_load_timm_backbone(pretrained=False)
        except Exception as exc:
            original_error = f"{self.backbone_type}: {exc}"
            self.backbone_load_error = original_error
            self._try_load_timm_backbone(pretrained=False, preserve_error=True)
            if self.backbone_load_error is None:
                self.backbone_load_error = original_error

    def _load_dino_backbone(self):
        hub_name = self.DINO_HUB_NAMES.get(self.backbone_type)
        if hub_name is None:
            raise ValueError(f"unsupported backbone_type: {self.backbone_type}")
        backbone = torch.hub.load("facebookresearch/dinov2", hub_name)
        if self.backbone_checkpoint_path:
            payload = torch.load(self.backbone_checkpoint_path, map_location="cpu")
            state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload
            backbone.load_state_dict(state_dict, strict=False)
        backbone_dim = 1024 if "vitl14" in hub_name else 768
        self._finalize_backbone(backbone, backbone_dim)

    def _try_load_timm_backbone(self, pretrained: bool = False,
                                preserve_error: bool = False):
        previous_error = self.backbone_load_error
        try:
            import timm
            backbone = timm.create_model(
                "vit_base_patch16_224", pretrained=pretrained,
                num_classes=0, global_pool="",
            )
            self._finalize_backbone(backbone, 768)
            if preserve_error:
                self.backbone_load_error = previous_error
        except Exception as exc:
            self.backbone_load_error = previous_error or f"timm fallback failed: {exc}"
            self.backbone = None
            self.backbone_dim = self.d_model
            self.backbone_proj = nn.Identity()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone is not None and self.backbone_freeze:
            self.backbone.eval()
        return self

    def _extract_backbone_tokens(self, flat: torch.Tensor) -> torch.Tensor:
        if self.backbone is None:
            return flat
        if self.backbone_freeze:
            with torch.no_grad():
                features = self.backbone(flat)
        else:
            features = self.backbone(flat)
        if isinstance(features, dict):
            features = features.get(
                "x_norm_patchtokens",
                features.get("tokens", features.get("last_hidden_state")),
            )
        if features.dim() == 2:
            features = features.unsqueeze(1)
        if features.shape[1] > 1 and self.backbone_type.startswith("dinov2"):
            return features
        if features.shape[1] > 1 and features.shape[1] != (flat.shape[-1] // 16) ** 2:
            return features[:, 1:]
        return features

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        B, N = images.shape[:2]
        flat = images.reshape(B * N, *images.shape[2:])
        features = self._extract_backbone_tokens(flat)
        features = self.backbone_proj(features)
        return features.view(B, N, features.shape[1], features.shape[2])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, N, 3, H, W] raw images OR [B, N, P, D] pre-extracted
        Returns:
            t1, t2_pointmap, t2_confidence, t3_evidence, t3_named, perception_summary
        """
        if self.use_backbone and x.dim() == 5:
            t1 = self.encode_images(x)
        else:
            t1 = x

        t2_pointmap = self.pointmap_head(t1)
        t2_confidence = torch.sigmoid(self.confidence_head(t1))
        pooled = t1.mean(dim=2)

        t3_named = {}
        t3_list = []
        for name in self.EVIDENCE_SIGNALS:
            sig = self.evidence_projectors[name](pooled)
            t3_named[name] = sig
            t3_list.append(sig)

        t3 = torch.stack(t3_list, dim=2)
        perception_summary = t1.mean(dim=(1, 2))

        return {
            "t1": t1,
            "t2_pointmap": t2_pointmap,
            "t2_confidence": t2_confidence,
            "t3_evidence": t3,
            "t3_named": t3_named,
            "perception_summary": perception_summary,
            "backbone_type": self.backbone_type,
            "backbone_load_error": self.backbone_load_error,
        }
'''
text = text[:start] + new + text[end:]
p.write_text(text)
