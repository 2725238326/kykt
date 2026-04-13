from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    value: str
    label: str
    description: str
    param_family: str
    source_types: tuple[str, ...]
    default_runner: str


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "dust3r": ModelSpec(
        value="dust3r",
        label="DUSt3R",
        description="图片对 / 多图三维重建",
        param_family="image_collection",
        source_types=("images",),
        default_runner="dust3r_runner.py",
    ),
    "mast3r": ModelSpec(
        value="mast3r",
        label="MASt3R",
        description="更强的静态多图匹配与三维重建",
        param_family="image_collection",
        source_types=("images",),
        default_runner="mast3r_runner.py",
    ),
    "monst3r": ModelSpec(
        value="monst3r",
        label="MonST3R",
        description="视频 / 帧序列动态三维重建",
        param_family="video_sequence",
        source_types=("video", "frames"),
        default_runner="monst3r_runner.py",
    ),
}


MODEL_OPTIONS = [
    {
        "value": spec.value,
        "label": spec.label,
        "description": spec.description,
    }
    for spec in MODEL_REGISTRY.values()
]


SOURCE_TYPE_OPTIONS = [
    {"value": "images", "label": "图片"},
    {"value": "video", "label": "视频"},
    {"value": "frames", "label": "帧序列"},
]


def get_model_spec(model: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[model]
    except KeyError as exc:
        raise KeyError(f"未知模型：{model}") from exc


def allowed_source_types(model: str) -> tuple[str, ...]:
    return get_model_spec(model).source_types


def default_runner_for(model: str) -> str:
    return get_model_spec(model).default_runner


def param_family_for(model: str) -> str:
    return get_model_spec(model).param_family
