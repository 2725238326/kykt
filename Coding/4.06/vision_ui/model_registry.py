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
    launch_blocker: str | None = None


@dataclass(frozen=True)
class ModelCatalogEntry:
    value: str
    label: str
    description: str
    family: str
    param_family: str
    source_types: tuple[str, ...]
    runner_status: str
    research_priority: int
    active_track: bool = True
    launch_blocker: str | None = None


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
    "spann3r": ModelSpec(
        value="spann3r",
        label="Spann3R",
        description="Spatial memory 全局点图重建",
        param_family="spann3r_sequence",
        source_types=("images", "frames"),
        default_runner="spann3r_runner.py",
        family="memory_global_pointmap",
        runner_status="smoke_ready",
        research_priority=95,
    ),
    "fast3r": ModelSpec(
        value="fast3r",
        label="Fast3R",
        description="长图集快速前馈三维重建",
        param_family="fast3r_collection",
        source_types=("images", "frames"),
        default_runner="fast3r_runner.py",
        family="large_image_collection",
        runner_status="smoke_ready_attention_fallback",
        research_priority=90,
    ),
}


MODEL_CATALOG: dict[str, ModelCatalogEntry] = {
    key: ModelCatalogEntry(
        value=spec.value,
        label=spec.label,
        description=spec.description,
        family=spec.family,
        param_family=spec.param_family,
        source_types=spec.source_types,
        runner_status=spec.runner_status,
        research_priority=spec.research_priority,
        active_track=spec.active_track,
        launch_blocker=spec.launch_blocker,
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
            param_family="spann3r_sequence",
            source_types=("images", "frames"),
            runner_status="smoke_ready",
            research_priority=95,
        ),
        "align3r": ModelCatalogEntry(
            value="align3r",
            label="Align3R",
            description="动态视频深度一致性与动态点云",
            family="video_depth_consistency",
            param_family="video_sequence",
            source_types=("video", "frames"),
            runner_status="env_blocked_curope",
            research_priority=94,
            launch_blocker="远端 Align3R 仍缺 curope / Depth Pro / Depth Anything V2 环境确认和官方 smoke run。",
        ),
        "fast3r": ModelCatalogEntry(
            value="fast3r",
            label="Fast3R",
            description="长图集快速前馈三维重建",
            family="large_image_collection",
            param_family="fast3r_collection",
            source_types=("images", "frames"),
            runner_status="smoke_ready_attention_fallback",
            research_priority=90,
        ),
        "cut3r": ModelCatalogEntry(
            value="cut3r",
            label="CUT3R",
            description="在线 / persistent-state 三维感知",
            family="streaming_state_reconstruction",
            param_family="streaming_sequence",
            source_types=("video", "frames", "images"),
            runner_status="env_blocked_curope",
            research_priority=88,
            launch_blocker="远端 CUT3R 仍缺 curope / gsplat / evo / open3d 环境确认、权重落位和官方 smoke run。",
        ),
        "pi3x": ModelCatalogEntry(
            value="pi3x",
            label="Pi3X",
            description="无参考视角通用视觉几何",
            family="general_visual_geometry",
            param_family="research_catalog",
            source_types=("images", "video", "frames"),
            runner_status="frontier_research",
            research_priority=55,
            active_track=False,
            launch_blocker="前沿预研条目，尚未定义 runner、部署合同和标准输出。",
        ),
        "zipmap": ModelCatalogEntry(
            value="zipmap",
            label="ZipMap",
            description="线性时间有状态三维重建",
            family="stateful_linear_reconstruction",
            param_family="research_catalog",
            source_types=("images", "video", "frames"),
            runner_status="frontier_research",
            research_priority=50,
            active_track=False,
            launch_blocker="前沿预研条目，尚未定义 runner、部署合同和标准输出。",
        ),
        "lingbot_map": ModelCatalogEntry(
            value="lingbot_map",
            label="LingBot-Map",
            description="流式几何上下文建图",
            family="streaming_mapping",
            param_family="research_catalog",
            source_types=("video", "frames"),
            runner_status="frontier_research",
            research_priority=45,
            active_track=False,
            launch_blocker="前沿预研条目，尚未定义 runner、部署合同和标准输出。",
        ),
    }
)


MODEL_OPTIONS = [
    {
        "value": spec.value,
        "label": spec.label,
        "description": spec.description,
        "param_family": spec.param_family,
        "family": spec.family,
        "source_types": list(spec.source_types),
        "runner_status": spec.runner_status,
        "research_priority": spec.research_priority,
        "active_track": spec.active_track,
        "runnable": True,
        "launch_blocker": spec.launch_blocker,
    }
    for spec in MODEL_REGISTRY.values()
]


MODEL_CATALOG_OPTIONS = [
    {
        "value": spec.value,
        "label": spec.label,
        "description": spec.description,
        "family": spec.family,
        "param_family": spec.param_family,
        "source_types": list(spec.source_types),
        "runner_status": spec.runner_status,
        "research_priority": spec.research_priority,
        "active_track": spec.active_track,
        "runnable": spec.value in MODEL_REGISTRY,
        "launch_blocker": spec.launch_blocker,
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
