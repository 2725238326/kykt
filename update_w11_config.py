from pathlib import Path
config = Path('/hdd3/kykt26/code/dream3r/dream3r/config.py')
text = config.read_text()
replacements = [
('''    "patch_size": 16,
    "use_backbone": False,
''', '''    "patch_size": 16,
    "use_backbone": False,
    "backbone_type": "dinov2_vitb14",
    "backbone_freeze": True,
    "backbone_checkpoint_path": "",
'''),
('''        "img_size": cfg["img_size"],
        "profile": cfg.get("profile", False),
''', '''        "img_size": cfg["img_size"],
        "backbone_type": cfg.get("backbone_type", "dinov2_vitb14"),
        "backbone_freeze": cfg.get("backbone_freeze", True),
        "backbone_checkpoint_path": cfg.get("backbone_checkpoint_path", ""),
        "profile": cfg.get("profile", False),
'''),
]
for old, new in replacements:
    if old not in text:
        raise SystemExit(f'missing config pattern {old!r}')
    text = text.replace(old, new)
config.write_text(text)

model = Path('/hdd3/kykt26/code/dream3r/dream3r/model.py')
text = model.read_text()
replacements = [
('''        use_backbone = c.get("use_backbone", False)
        img_size     = c.get("img_size", 224)
''', '''        use_backbone = c.get("use_backbone", False)
        img_size     = c.get("img_size", 224)
        backbone_type = c.get("backbone_type", "dinov2_vitb14")
        backbone_freeze = c.get("backbone_freeze", True)
        backbone_checkpoint_path = c.get("backbone_checkpoint_path", "")
'''),
('''            d_model=d_model, n_evidence=n_evidence, d_evidence=d_evidence,
            img_size=img_size, use_backbone=use_backbone,
        )
''', '''            d_model=d_model, n_evidence=n_evidence, d_evidence=d_evidence,
            img_size=img_size, use_backbone=use_backbone,
            backbone_type=backbone_type, backbone_freeze=backbone_freeze,
            backbone_checkpoint_path=backbone_checkpoint_path,
        )
'''),
('''        "use_backbone": False, "img_size": 224,
    },
''', '''        "use_backbone": False, "img_size": 224,
        "backbone_type": "none", "backbone_freeze": True,
        "backbone_checkpoint_path": "",
    },
'''),
('''        "use_backbone": False, "img_size": 224,
    },
    "small_vit": {
''', '''        "use_backbone": False, "img_size": 224,
        "backbone_type": "none", "backbone_freeze": True,
        "backbone_checkpoint_path": "",
    },
    "small_vit": {
'''),
('''        "use_backbone": True, "img_size": 224,
    },
''', '''        "use_backbone": True, "img_size": 224,
        "backbone_type": "dinov2_vitb14", "backbone_freeze": True,
        "backbone_checkpoint_path": "",
    },
'''),
]
for old, new in replacements:
    if old not in text:
        raise SystemExit(f'missing model pattern {old!r}')
    text = text.replace(old, new, 1)
model.write_text(text)
