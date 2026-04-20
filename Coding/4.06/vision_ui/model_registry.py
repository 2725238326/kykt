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
    family: str
    runner_status: str
    research_priority: int
    active_track: bool = True


@dataclass(frozen=True)
class ModelCatalogEntry:
    value: str
    label: str
    description: str
    family: str
    source_types: tuple[str, ...]
    runner_status: str
    research_priority: int
    active_track: bool = True


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "dust3r": ModelSpec(
        value="dust3r",
        label="DUSt3R",
        description="图片对 / 多图三维重建",
        param_family="image_collection",
        source_types=("images",),
        default_runner="dust3r_runner.py",
        family="pairwise_pointmap",
        runner_status="baseline",
        research_priority=90,
        active_track=False,
    ),
    "mast3r": ModelSpec(
        value="mast3r",
        label="MASt3R",
        description="更强的静态多图匹配与三维重建",
        param_family="image_collection",
        source_types=("images",),
        default_runner="mast3r_runner.py",
        family="static_matching_reconstruction",
        runner_status="validated_smoke",
        research_priority=100,
    ),
    "monst3r": ModelSpec(
        value="monst3r",
        label="MonST3R",
        description="视频 / 帧序列动态三维重建",
        param_family="video_sequence",
        source_types=("video", "frames"),
        default_runner="monst3r_runner.py",
        family="video_dynamic_reconstruction",
        runner_status="validated_standard_sample",
        research_priority=100,
    ),
}


MODEL_CATALOG: dict[str, ModelCatalogEntry] = {
    key: ModelCatalogEntry(
        value=spec.value,
        label=spec.label,
        description=spec.description,
        family=spec.family,
        source_types=spec.source_types,
        runner_status=spec.runner_status,
        research_priority=spec.research_priority,
        active_track=spec.active_track,
    )
    for key, spec in MODEL_REGISTRY.items()
}

MODEL_CATALOG.update(
    {
        "spann3r": ModelCatalogEntry(
            value="spann3r",
            label="Spann3R",
            description="Spatial memory 全局点图重建",
            family="memory_global_pointmap",
            source_types=("images", "frames"),
            runner_status="planned",
            research_priority=95,
        ),
        "align3r": ModelCatalogEntry(
            value="align3r",
            label="Align3R",
            description="动态视频深度一致性与动态点云",
            family="video_depth_consistency",
            source_types=("video", "frames"),
            runner_status="planned",
            research_priority=94,
        ),
        "fast3r": ModelCatalogEntry(
            value="fast3r",
            label="Fast3R",
            description="长图集快速前馈三维重建",
            family="large_image_collection",
            source_types=("images", "frames"),
            runner_status="planned",
            research_priority=90,
        ),
        "cut3r": ModelCatalogEntry(
            value="cut3r",
            label="CUT3R",
            description="在线 / persistent-state 三维感知",
            family="streaming_state_reconstruction",
            source_types=("video", "frames", "images"),
            runner_status="planned",
            research_priority=88,
        ),
        "pi3x": ModelCatalogEntry(
            value="pi3x",
            label="Pi3X",
            description="无参考视角通用视觉几何",
            family="general_visual_geometry",
            source_types=("images", "video", "frames"),
            runner_status="frontier_research",
            research_priority=55,
            active_track=False,
        ),
        "zipmap": ModelCatalogEntry(
            value="zipmap",
            label="ZipMap",
            description="线性时间有状态三维重建",
            family="stateful_linear_reconstruction",
            source_types=("images", "video", "frames"),
            runner_status="frontier_research",
            research_priority=50,
            active_track=False,
        ),
        "lingbot_map": ModelCatalogEntry(
            value="lingbot_map",
            label="LingBot-Map",
            description="流式几何上下文建图",
            family="streaming_mapping",
            source_types=("video", "frames"),
            runner_status="frontier_research",
            research_priority=45,
            active_track=False,
        ),
    }
)


MODEL_OPTIONS = [
    {
        "value": spec.value,
        "label": spec.label,
        "description": spec.description,
        "family": spec.family,
        "runner_status": spec.runner_status,
        "research_priority": spec.research_priority,
        "active_track": spec.active_track,
    }
    for spec in MODEL_REGISTRY.values()
]


MODEL_CATALOG_OPTIONS = [
    {
        "value": spec.value,
        "label": spec.label,
        "description": spec.description,
        "family": spec.family,
        "source_types": list(spec.source_types),
        "runner_status": spec.runner_status,
        "research_priority": spec.research_priority,
        "active_track": spec.active_track,
        "runnable": spec.value in MODEL_REGISTRY,
    }
    for spec in sorted(MODEL_CATALOG.values(), key=lambda item: (-item.research_priority, item.value))
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
