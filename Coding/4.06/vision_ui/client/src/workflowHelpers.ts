import type { JobsListPayload } from "./types";
import {
  defaultDust3rParams,
  defaultFast3rParams,
  defaultMonst3rParams
} from "./appConfig";
import type { PresetKey } from "./appConfig";
import {
  sourceTypeLabel,
  statusLabel,
  statusModelLabel
} from "./displayHelpers";
import type { ModelCatalogItem } from "./displayHelpers";

export type CreateParamMode = "image_collection" | "video_sequence" | "spann3r_sequence" | "fast3r_collection" | "catalog";

export type FormState = {
  model: string;
  source_type: string;
  notes: string;
} & typeof defaultDust3rParams &
  typeof defaultMonst3rParams &
  typeof defaultFast3rParams;

export type BatchJobAction = "dispatch" | "retry" | "cancel";
export type JobListItem = JobsListPayload["jobs"][number];

export function isParamFieldKey(key: keyof FormState) {
  return key in defaultDust3rParams || key in defaultMonst3rParams || key in defaultFast3rParams;
}

export function fallbackParamFamilyForModel(model: string) {
  const localRegistry: Record<string, string> = {
    dust3r: "image_collection",
    mast3r: "image_collection",
    monst3r: "video_sequence",
    spann3r: "spann3r_sequence",
    fast3r: "fast3r_collection",
    align3r: "video_sequence",
    cut3r: "streaming_sequence",
    pi3x: "research_catalog",
    zipmap: "research_catalog",
    lingbot_map: "research_catalog"
  };
  return localRegistry[model] ?? "image_collection";
}

export function fallbackSourceTypesForModel(model: string) {
  const localRegistry: Record<string, string[]> = {
    dust3r: ["images"],
    mast3r: ["images"],
    monst3r: ["video", "frames"],
    spann3r: ["images", "frames"],
    align3r: ["video", "frames"],
    fast3r: ["images", "frames"],
    cut3r: ["video", "frames", "images"],
    pi3x: ["images", "video", "frames"],
    zipmap: ["images", "video", "frames"],
    lingbot_map: ["video", "frames"]
  };
  return localRegistry[model] ?? ["images"];
}

export function createParamModeForFamily(paramFamily: string | undefined): CreateParamMode {
  switch (paramFamily) {
    case "image_collection":
      return "image_collection";
    case "video_sequence":
      return "video_sequence";
    case "spann3r_sequence":
      return "spann3r_sequence";
    case "fast3r_collection":
      return "fast3r_collection";
    default:
      return "catalog";
  }
}

export function getCreateParamMode(model: string, catalog?: ModelCatalogItem[]): CreateParamMode {
  const catalogItem = catalog?.find((item) => item.value === model);
  if (catalogItem && !catalogItem.runnable) {
    return "catalog";
  }
  return createParamModeForFamily(catalogItem?.param_family ?? fallbackParamFamilyForModel(model));
}

export function getParamDefaultsForMode(mode: CreateParamMode) {
  switch (mode) {
    case "video_sequence":
      return defaultMonst3rParams;
    case "fast3r_collection":
      return defaultFast3rParams;
    case "spann3r_sequence":
    case "catalog":
      return {};
    case "image_collection":
    default:
      return defaultDust3rParams;
  }
}

export function buildDust3rPreset(preset: PresetKey, fileCount: number): typeof defaultDust3rParams {
  const sceneGraph = fileCount > 6 ? "swin-5" : "complete";
  switch (preset) {
    case "quick":
      return {
        image_size: "224",
        scene_graph: "complete",
        niter: "150",
        lr: "0.01",
        batch_size: "1",
        max_points: "100000",
        match_viz_count: "20"
      };
    case "enhanced":
      return {
        image_size: "512",
        scene_graph: sceneGraph,
        niter: "500",
        lr: "0.005",
        batch_size: "1",
        max_points: "500000",
        match_viz_count: "100"
      };
    case "standard":
    default:
      return {
        image_size: "512",
        scene_graph: sceneGraph,
        niter: "300",
        lr: "0.01",
        batch_size: "1",
        max_points: "250000",
        match_viz_count: "50"
      };
  }
}

export function buildMonst3rPreset(preset: PresetKey): typeof defaultMonst3rParams {
  switch (preset) {
    case "quick":
      return {
        image_size: "224",
        batch_size: "1",
        fps: "0",
        num_frames: "24",
        not_batchify: "true",
        real_time: "false",
        window_wise: "false",
        window_size: "16",
        window_overlap_ratio: "0.25"
      };
    case "enhanced":
      return {
        image_size: "512",
        batch_size: "1",
        fps: "0",
        num_frames: "72",
        not_batchify: "true",
        real_time: "false",
        window_wise: "true",
        window_size: "32",
        window_overlap_ratio: "0.5"
      };
    case "standard":
    default:
      return {
        image_size: "512",
        batch_size: "1",
        fps: "0",
        num_frames: "48",
        not_batchify: "true",
        real_time: "false",
        window_wise: "false",
        window_size: "24",
        window_overlap_ratio: "0.5"
      };
  }
}

export function buildFast3rPreset(preset: PresetKey): typeof defaultFast3rParams {
  switch (preset) {
    case "quick":
      return {
        image_size: "224",
        max_points: "100000"
      };
    case "enhanced":
      return {
        image_size: "512",
        max_points: "500000"
      };
    case "standard":
    default:
      return {
        image_size: "512",
        max_points: "250000"
      };
  }
}

export function presetLabel(preset: PresetKey | null) {
  switch (preset) {
    case "quick":
      return "快速";
    case "enhanced":
      return "增强";
    case "standard":
      return "标准";
    default:
      return "自定义";
  }
}

export function allowedSourceTypesForModel(model: string, catalog?: ModelCatalogItem[]) {
  const fromCatalog = catalog?.find((item) => item.value === model)?.source_types ?? [];
  if (fromCatalog.length > 0) {
    return fromCatalog;
  }
  return fallbackSourceTypesForModel(model);
}

export function inputHint(model: string, sourceType: string) {
  if (model === "monst3r" && sourceType === "video") {
    return "上传 1 个视频文件";
  }
  if (model === "monst3r") {
    return "上传连续帧图片，3 张以上";
  }
  if (model === "spann3r") {
    return sourceType === "frames" ? "上传连续帧图片，3 到 12 张" : "上传多视角图片，3 张以上";
  }
  if (model === "fast3r") {
    return sourceType === "frames" ? "上传关键帧或连续帧，4 张以上" : "上传多视角图片，4 到 12 张";
  }
  return model === "mast3r" ? "上传 2 张或更多同场景图片" : "上传 2 张或更多图片";
}

export function buildCreateGuidance(model: string, sourceType: string, fileCount: number) {
  const modelLabel = statusModelLabel(model);
  const inputLabel = sourceTypeLabel(sourceType);
  if (sourceType === "video") {
    return [`${modelLabel} · ${inputLabel}`, fileCount === 1 ? "输入已就绪。" : "需要 1 个视频文件。"];
  }
  return [
    `${modelLabel} · ${inputLabel}`,
    fileCount >= 2 ? "输入已就绪。" : "至少需要 2 张图片或帧。"
  ];
}

export function buildCreateReadiness(
  serviceReady: boolean,
  modelItem: ModelCatalogItem | null,
  allowedSourceTypes: string[],
  model: string,
  sourceType: string,
  fileCount: number,
  imageCount: number,
  videoCount: number
) {
  const inputReady =
    sourceType === "video"
      ? fileCount === 1 && videoCount === 1
      : sourceType === "frames"
        ? imageCount >= 2
        : imageCount >= 2;
  const modelReady = modelItem?.runnable ?? true;
  return [
    {
      label: "服务",
      value: serviceReady ? "Ready" : "Waiting",
      tone: serviceReady ? "ready" : "blocked"
    },
    {
      label: "模型",
      value: modelReady ? statusModelLabel(model) : "Blocked",
      tone: modelReady ? "ready" : "blocked"
    },
    {
      label: "输入",
      value: inputReady ? `${fileCount} 个` : "待补",
      tone: inputReady ? "ready" : "partial"
    },
    {
      label: "来源",
      value: sourceTypeLabel(sourceType),
      tone: allowedSourceTypes.includes(sourceType) ? "ready" : "blocked"
    }
  ];
}

export function buildParamPanelIntro(mode: CreateParamMode) {
  switch (mode) {
    case "video_sequence":
      return "视频序列参数档位。";
    case "fast3r_collection":
      return "Fast3R 分辨率与点数上限。";
    case "spann3r_sequence":
      return "Spann3R 固定序列参数。";
    case "catalog":
      return "目录模型暂不暴露参数。";
    case "image_collection":
    default:
      return "图像集合参数档位。";
  }
}

export function buildCreateLaunchHeadline(serviceReady: boolean, fileCount: number, modelBlocker: string | null) {
  if (!serviceReady) {
    return "等待本地服务";
  }
  if (modelBlocker) {
    return "模型暂不可创建";
  }
  if (fileCount === 0) {
    return "先选择输入文件";
  }
  return "可以创建任务";
}

export function buildCreateLaunchMessage(serviceReady: boolean, model: string, sourceType: string, fileCount: number, modelBlocker: string | null) {
  if (!serviceReady) {
    return "本地服务就绪后才能提交，当前配置会保留在页面上。";
  }
  if (modelBlocker) {
    return modelBlocker;
  }
  if (fileCount === 0) {
    return "先把文件放进 staging 区，再按当前模型和输入来源发起任务。";
  }
  return `${statusModelLabel(model)} / ${sourceTypeLabel(sourceType)} / ${fileCount} 个输入文件。`;
}

export function buildCaptureChecklist(model: string, sourceType: string, fileCount: number) {
  const target = sourceType === "video" ? "1 个视频" : "2 张以上图片或帧";
  const ready = sourceType === "video" ? fileCount === 1 : fileCount >= 2;
  const body = ready ? `已选择 ${fileCount} 个文件。` : `需要 ${target}。`;
  return [
    {
      title: "输入要求",
      body: target
    },
    {
      title: "来源",
      body: sourceTypeLabel(sourceType)
    },
    {
      title: "当前文件",
      body
    }
  ];
}

export function buildAdvisorChecklist(advisorReady: boolean) {
  return [
    {
      title: "当前状态",
      body: advisorReady ? "已配置，可生成草稿。" : "未就绪，先补配置。"
    },
    {
      title: "定位",
      body: "现阶段只作为辅助草稿。"
    },
    {
      title: "后续",
      body: "评估框架会单独重构。"
    }
  ];
}

export function buildSystemChecklist() {
  return [
    {
      title: "任务流",
      body: "文件与新建 -> 运行与结果 -> Matrix。"
    },
    {
      title: "评估",
      body: "辅助评估先保留为草稿入口。"
    },
    {
      title: "服务",
      body: "本地异常时从顶部或系统页恢复。"
    }
  ];
}

export function buildActionMessage(action: string, jobId: string) {
  switch (action) {
    case "dispatch":
      return `任务 ${jobId} 已开始调度。`;
    case "retry":
      return `任务 ${jobId} 已重新进入调度流程。`;
    case "duplicate":
      return `已复制出新的任务 ${jobId}。`;
    case "cancel":
      return `任务 ${jobId} 已请求取消。`;
    case "advisor":
      return `任务 ${jobId} 的辅助评估已更新。`;
    default:
      return `任务 ${jobId} 已更新。`;
  }
}

export function batchActionLabel(action: BatchJobAction) {
  switch (action) {
    case "dispatch":
      return "运行";
    case "retry":
      return "重试";
    case "cancel":
      return "取消";
    default:
      return "操作";
  }
}

export function isAdvisorSuggested(status: string) {
  return status === "finished" || status === "failed" || status === "cancelled";
}

export function buildModelLaunchBlocker(item: ModelCatalogItem | null) {
  if (!item || item.runnable) {
    return null;
  }
  return item.launch_blocker ?? `${item.label} 还没有接入可派发 runner，先保留在部署/研究目录。`;
}

export function buildMatrixModelConstraint(item: ModelCatalogItem) {
  if (!item.runnable) {
    return {
      tone: "blocked",
      label: "Catalog-only",
      detail: item.launch_blocker ?? "目录模型暂不进入创建队列。"
    };
  }
  if (item.runner_status === "smoke_ready_attention_fallback" || item.runner_status === "validated_smoke_attention_fallback") {
    return {
      tone: "partial",
      label: "Fallback",
      detail: "当前走 attention fallback，先小样例确认速度。"
    };
  }
  if (item.runner_status === "baseline") {
    return {
      tone: "partial",
      label: "Baseline",
      detail: "基座保留线，优先作为静态参考。"
    };
  }
  return null;
}

export function canDispatchJobStatus(status: string) {
  return status === "draft" || status === "ready" || status === "failed" || status === "cancelled";
}

export function matchesJobQuery(item: JobListItem, normalizedQuery: string) {
  if (!normalizedQuery) {
    return true;
  }
  const haystack = [
    item.job.job_id,
    item.job.model,
    statusLabel(item.job.status),
    sourceTypeLabel(item.job.source_type),
    item.phase_display.label,
    item.job.progress_message ?? ""
  ]
    .join(" ")
    .toLowerCase();
  return haystack.includes(normalizedQuery);
}

export function buildOverviewHeadline(job: JobListItem | null, runningCount: number, attentionCount: number) {
  if (!job) {
    return "创建一条样例任务。";
  }
  if (job.job.status === "running") {
    return runningCount > 1 ? `运行中 ${runningCount} 条。` : "任务运行中。";
  }
  if (attentionCount > 0) {
    return "有待处理任务。";
  }
  if (job.job.status === "finished") {
    return "结果已回传。";
  }
  return "当前空闲。";
}

export function buildOverviewMessage(job: JobListItem | null, runningCount: number, attentionCount: number) {
  if (!job) {
    return "输入、远端执行、结果回传会在任务中心汇总。";
  }
  if (job.job.status === "running") {
    return `${statusModelLabel(job.job.model)} · ${job.phase_display.label}`;
  }
  if (attentionCount > 0) {
    return "待处理筛选会集中显示失败和取消任务。";
  }
  if (job.job.status === "finished") {
    return `${statusModelLabel(job.job.model)} · 核心结果可检查。`;
  }
  return runningCount > 0 ? "后台仍有其他任务在执行。" : "可发起下一条任务。";
}
