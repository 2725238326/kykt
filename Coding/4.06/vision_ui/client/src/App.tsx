import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type {
  AdvisorConfig,
  AdvisorReport,
  AdvisorStatus,
  BackendStatusPayload,
  BootstrapPayload,
  DeploymentStatusPayload,
  EvaluationPayload,
  JobPayload,
  JobsListPayload,
  ResultSummary,
  SamplesPayload
} from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE || "http://127.0.0.1:8765").replace(/\/$/, "");

const DEFAULT_BOOTSTRAP: BootstrapPayload = {
  summary: { total: 0, running: 0, finished: 0, failed: 0, cancelled: 0 },
  delivery_gaps: [],
  server: {
    alias: "KYKT-UI",
    host: "172.17.140.97",
    user: "kykt26",
    port: 22,
    remote_root: "/hdd3/kykt26"
  },
  models: [
    {
      value: "dust3r",
      label: "DUSt3R",
      description: "图片对 / 多图三维重建"
    },
    {
      value: "mast3r",
      label: "MASt3R",
      description: "更强的静态多图匹配与三维重建"
    },
    {
      value: "monst3r",
      label: "MonST3R",
      description: "视频 / 帧序列动态三维重建"
    }
  ],
  source_types: [
    { value: "images", label: "图片" },
    { value: "frames", label: "帧序列" },
    { value: "video", label: "视频" }
  ]
};

const defaultDust3rParams = {
  image_size: "512",
  scene_graph: "complete",
  niter: "300",
  lr: "0.01",
  batch_size: "1",
  max_points: "250000",
  match_viz_count: "50"
};

const defaultMonst3rParams = {
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

type ParamChoice = {
  value: string;
  label: string;
  note: string;
};

type PresetKey = "quick" | "standard" | "enhanced";
type PresetModel = "dust3r" | "mast3r" | "monst3r";

type PresetDescriptor = {
  key: PresetKey;
  label: string;
  note: string;
};

const dust3rParamChoices: Record<keyof typeof defaultDust3rParams, ParamChoice[]> = {
  image_size: [
    { value: "512", label: "512（标准推荐）", note: "正式样例优先用这一档，细节和稳定性更平衡。" },
    { value: "384", label: "384（中速）", note: "想稍微提速又不想直接降到最低时可用。" },
    { value: "224", label: "224（快速摸底）", note: "只适合先验流程，不建议拿来做最终展示。" }
  ],
  scene_graph: [
    { value: "complete", label: "complete（2 到 6 张推荐）", note: "少量图片时最稳，配对最完整。" },
    { value: "swin-5", label: "swin-5（6 张以上推荐）", note: "图片较多时更省配对成本，避免 complete 过重。" }
  ],
  niter: [
    { value: "150", label: "150（快速）", note: "适合先看大概效果，细节优化较少。" },
    { value: "300", label: "300（基线推荐）", note: "最适合作为第一版正式实验参数。" },
    { value: "500", label: "500（精细）", note: "更适合重点样例，耗时会明显增加。" }
  ],
  lr: [
    { value: "0.005", label: "0.005（更稳）", note: "想保守一点时用这一档。" },
    { value: "0.01", label: "0.01（标准推荐）", note: "当前默认最合适，先从这里起步。" },
    { value: "0.02", label: "0.02（更激进）", note: "只在你想试更快收敛时再用。" }
  ],
  batch_size: [
    { value: "1", label: "1（稳妥推荐）", note: "最稳，不容易因为显存或负载出额外问题。" },
    { value: "2", label: "2（显存充足）", note: "想提速且机器有余量时可以尝试。" }
  ],
  max_points: [
    { value: "100000", label: "100000（快速）", note: "点更少，导出更轻，适合快速筛样例。" },
    { value: "250000", label: "250000（基线推荐）", note: "当前最均衡，先用这一档就好。" },
    { value: "500000", label: "500000（细节优先）", note: "更适合重点样例，文件和开销都会变大。" }
  ],
  match_viz_count: [
    { value: "0", label: "0（不画匹配线）", note: "只关心点云结果时可以关掉。" },
    { value: "20", label: "20（简洁）", note: "适合快速看有没有基本匹配。" },
    { value: "50", label: "50（基线推荐）", note: "展示和排查都比较合适。" },
    { value: "100", label: "100（更密）", note: "想看更丰富的匹配关系时再开高。" }
  ]
};

const monst3rParamChoices: Record<keyof typeof defaultMonst3rParams, ParamChoice[]> = {
  image_size: [
    { value: "512", label: "512（正式样例推荐）", note: "明天做正式视频样例时优先用这一档。" },
    { value: "224", label: "224（快速验链路）", note: "只适合先验证能不能跑通，不适合最终展示。" }
  ],
  batch_size: [
    { value: "1", label: "1（稳妥推荐）", note: "当前最稳，先别急着往上加。" },
    { value: "2", label: "2（显存足够再试）", note: "想提速且显存够时再考虑。" }
  ],
  fps: [
    { value: "0", label: "0（自动/原节奏推荐）", note: "最省心，主要由最大帧数控制总量。" },
    { value: "2", label: "2（更省）", note: "视频偏长时，先压到这一档更容易控住规模。" },
    { value: "4", label: "4（常规抽帧）", note: "想更均匀地抽样视频内容时可用。" },
    { value: "8", label: "8（高动作场景）", note: "只有运动较快时才建议提到这么高。" }
  ],
  num_frames: [
    { value: "24", label: "24（快速验链路）", note: "只看能不能跑通时可选。" },
    { value: "48", label: "48（基线推荐）", note: "正式第一版样例优先用这一档。" },
    { value: "72", label: "72（增强）", note: "效果不错后再往上加，适合更完整的弧线运动。" },
    { value: "96", label: "96（长序列）", note: "更重，只建议对重点视频使用。" }
  ],
  not_batchify: [
    { value: "true", label: "开启（稳妥推荐）", note: "更适合我们当前这套环境，先保持开启。" },
    { value: "false", label: "关闭（速度优先）", note: "只有你确认资源富余时再尝试关闭。" }
  ],
  real_time: [
    { value: "false", label: "关闭（离线质量推荐）", note: "做实验和交差样例时应保持关闭。" },
    { value: "true", label: "开启（演示模式）", note: "偏演示用途，不建议拿来做正式结果。" }
  ],
  window_wise: [
    { value: "false", label: "关闭（短视频推荐）", note: "短视频或第一版样例先别开。" },
    { value: "true", label: "开启（长序列推荐）", note: "视频更长或想更稳时再打开。" }
  ],
  window_size: [
    { value: "16", label: "16（更轻）", note: "短窗口，开销更小。" },
    { value: "24", label: "24（基线推荐）", note: "当前最合适，和 48 到 72 帧搭配比较稳。" },
    { value: "32", label: "32（更长序列）", note: "适合更长视频，但耗时更高。" }
  ],
  window_overlap_ratio: [
    { value: "0.25", label: "0.25（更快）", note: "重叠更少，速度更快。" },
    { value: "0.5", label: "0.5（基线推荐）", note: "目前最平衡，先用这一档。" },
    { value: "0.75", label: "0.75（更稳）", note: "更适合重点样例，代价是更慢。" }
  ]
};

const presetDescriptors: PresetDescriptor[] = [
  { key: "quick", label: "快速", note: "先验链路、快速出结果" },
  { key: "standard", label: "标准", note: "正式样例首选基线" },
  { key: "enhanced", label: "增强", note: "重点样例，质量优先" }
];

type FormState = {
  model: string;
  source_type: string;
  notes: string;
} & typeof defaultDust3rParams &
  typeof defaultMonst3rParams;

type ServiceState = "starting" | "ready" | "degraded";
type JobFilter = "all" | "running" | "attention" | "finished";
type WorkspaceTab = "overview" | "create" | "jobs" | "advisor" | "system";
type JobListItem = JobsListPayload["jobs"][number];
type OutputItem = JobPayload["outputs"][number];
type OutputSection = {
  key: string;
  title: string;
  description: string;
  accent: "blue" | "green" | "gold" | "slate";
  defaultOpen: boolean;
  items: OutputItem[];
};
type SampleMatrixJob = NonNullable<SamplesPayload["job_matrix"]>["rows"][number]["jobs_by_model"][string];
type SampleManifestEntry = NonNullable<SamplesPayload["manifest"]["samples"]>[number];
type SampleMatrixSortKey = "manifest" | "completion" | "score" | "attention";
type SampleMatrixFilterKey = "all" | "attention" | "running" | "unfinished";
type SampleMatrixRowView = {
  sample: SampleManifestEntry;
  rowIndex: number;
  jobsByModel: Record<string, SampleMatrixJob>;
  requiredModels: string[];
  requiredModelSet: Set<string>;
  compareModels: string[];
  rowStats: ReturnType<typeof summarizeSampleMatrixRow>;
  rowScore: ReturnType<typeof summarizeMatrixRowScore>;
};
type ModelCatalogItem = NonNullable<BootstrapPayload["model_catalog"]>[number];
type PreviewAsset = {
  url: string;
  name: string;
  kind: "image" | "video";
  note?: string;
};

const workspaceTabs: Array<{ key: WorkspaceTab; label: string; note: string }> = [
  { key: "overview", label: "工作台", note: "先看全局状态和焦点任务" },
  { key: "create", label: "文件与新建", note: "整理输入并创建任务" },
  { key: "jobs", label: "运行与结果", note: "筛选任务、跟进状态、查看产物" },
  { key: "advisor", label: "AI 评估", note: "让 AI 诊断结果并生成建议" },
  { key: "system", label: "帮助与系统", note: "查看服务、远端目标和使用提示" }
];

function App() {
  const [bootstrap, setBootstrap] = useState<BootstrapPayload | null>(null);
  const [jobs, setJobs] = useState<JobsListPayload["jobs"]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<JobPayload | null>(null);
  const [samplesPayload, setSamplesPayload] = useState<SamplesPayload | null>(null);
  const [samplesError, setSamplesError] = useState<string | null>(null);
  const [deploymentStatus, setDeploymentStatus] = useState<DeploymentStatusPayload | null>(null);
  const [deploymentError, setDeploymentError] = useState<string | null>(null);
  const [deploymentLoading, setDeploymentLoading] = useState(false);
  const [activeWorkspace, setActiveWorkspace] = useState<WorkspaceTab>("overview");
  const [backendStatus, setBackendStatus] = useState<BackendStatusPayload | null>(null);
  const [serviceState, setServiceState] = useState<ServiceState>("starting");
  const [serviceMessage, setServiceMessage] = useState("正在准备本地服务...");
  const [submitting, setSubmitting] = useState(false);
  const [actionKey, setActionKey] = useState<string | null>(null);
  const [jobFilter, setJobFilter] = useState<JobFilter>("all");
  const [jobQuery, setJobQuery] = useState("");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [infoMessage, setInfoMessage] = useState<string | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [recoveringService, setRecoveringService] = useState(false);
  const [savingEvaluation, setSavingEvaluation] = useState(false);
  const [previewAsset, setPreviewAsset] = useState<PreviewAsset | null>(null);
  const [advisorModalOpen, setAdvisorModalOpen] = useState(false);
  const [advisorConfigLoading, setAdvisorConfigLoading] = useState(false);
  const [advisorConfigSaving, setAdvisorConfigSaving] = useState(false);
  const [advisorForm, setAdvisorForm] = useState({
    enabled: false,
    base_url: "",
    api_key: "",
    model: "gpt-4o-mini",
    has_api_key: false,
  });
  const [activePresets, setActivePresets] = useState<Record<PresetModel, PresetKey | null>>({
    dust3r: "standard",
    mast3r: "standard",
    monst3r: "standard"
  });
  const recoveryInFlightRef = useRef(false);
  const [formState, setFormState] = useState<FormState>({
    model: "dust3r",
    source_type: "images",
    notes: "",
    ...defaultDust3rParams,
    ...defaultMonst3rParams
  });

  const bootstrapData = bootstrap ?? DEFAULT_BOOTSTRAP;
  const advisorState: AdvisorStatus = bootstrapData.advisor ?? {
    enabled: false,
    configured: false,
    base_url: "",
    model: "",
    has_api_key: false,
    message: "AI 评估尚未配置。"
  };
  const serviceReady = serviceState === "ready";
  const advisorReady = advisorState.enabled && advisorState.configured;
  const isImageCollectionModel = formState.model === "dust3r" || formState.model === "mast3r";
  const isMonst3r = formState.model === "monst3r";
  const selectedModel = useMemo(
    () => bootstrapData.models.find((item) => item.value === formState.model),
    [bootstrapData.models, formState.model]
  );
  const modelCatalog = useMemo<ModelCatalogItem[]>(
    () =>
      samplesPayload?.model_catalog ??
      bootstrapData.model_catalog ??
      bootstrapData.models.map((item) => ({
        value: item.value,
        label: item.label,
        description: item.description,
        family: item.family ?? "integrated",
        source_types: [],
        runner_status: item.runner_status ?? "integrated",
        research_priority: item.research_priority ?? 0,
        active_track: item.active_track ?? true,
        runnable: true
      })),
    [bootstrapData.model_catalog, bootstrapData.models, samplesPayload?.model_catalog]
  );
  const activeModelCatalog = useMemo(
    () => modelCatalog.filter((item) => item.active_track),
    [modelCatalog]
  );
  const deferredModelCatalog = useMemo(
    () => modelCatalog.filter((item) => !item.active_track),
    [modelCatalog]
  );
  const selectedListJob = useMemo(
    () => (selectedJobId ? jobs.find((item) => item.job.job_id === selectedJobId) ?? null : null),
    [jobs, selectedJobId]
  );
  const runningJobs = useMemo(() => jobs.filter((item) => item.job.status === "running"), [jobs]);
  const queuedJobs = useMemo(
    () => jobs.filter((item) => item.job.status === "draft" || item.job.status === "ready"),
    [jobs]
  );
  const hasRunningJobs = runningJobs.length > 0;
  const attentionJobs = useMemo(
    () => jobs.filter((item) => item.job.status === "failed" || item.job.status === "cancelled"),
    [jobs]
  );
  const finishedJobs = useMemo(() => jobs.filter((item) => item.job.status === "finished"), [jobs]);
  const focusJob = useMemo(
    () => selectedListJob ?? runningJobs[0] ?? attentionJobs[0] ?? jobs[0] ?? null,
    [selectedListJob, runningJobs, attentionJobs, jobs]
  );
  const activeJob = selectedJob ?? null;
  const advisorCandidateCount = useMemo(
    () => jobs.filter((item) => isAdvisorSuggested(item.job.status)).length,
    [jobs]
  );
  const normalizedJobQuery = jobQuery.trim().toLowerCase();
  const filteredJobs = useMemo(() => {
    const baseJobs: JobListItem[] = (() => {
      switch (jobFilter) {
        case "running":
          return runningJobs;
        case "attention":
          return attentionJobs;
        case "finished":
          return finishedJobs;
        default:
          return jobs;
      }
    })();
    if (!normalizedJobQuery) {
      return baseJobs;
    }
    return baseJobs.filter((item) => matchesJobQuery(item, normalizedJobQuery));
  }, [attentionJobs, finishedJobs, jobFilter, jobs, normalizedJobQuery, runningJobs]);
  const selectedFilteredIndex = useMemo(
    () => (selectedJobId ? filteredJobs.findIndex((item) => item.job.job_id === selectedJobId) : -1),
    [filteredJobs, selectedJobId]
  );
  const selectedFilteredModelCount = useMemo(() => {
    if (selectedFilteredIndex < 0) {
      return 0;
    }
    const selectedModel = filteredJobs[selectedFilteredIndex]?.job.model;
    if (!selectedModel) {
      return 0;
    }
    return filteredJobs.filter((item) => item.job.model === selectedModel).length;
  }, [filteredJobs, selectedFilteredIndex]);
  const canDispatchListJob = selectedListJob ? canDispatchJobStatus(selectedListJob.job.status) : false;
  const canRetryListJob = selectedListJob ? selectedListJob.job.status !== "running" : false;
  const canCancelListJob = selectedListJob ? selectedListJob.job.status === "running" : false;
  const filteredJobIds = useMemo(() => filteredJobs.map((item) => item.job.job_id), [filteredJobs]);
  const jobLaneCards = useMemo(
    () => [
      { key: "queue", label: "待派发", jobs: queuedJobs, filter: "all" as JobFilter, tone: "queue" as const },
      { key: "running", label: "运行中", jobs: runningJobs, filter: "running" as JobFilter, tone: "running" as const },
      { key: "attention", label: "待处理", jobs: attentionJobs, filter: "attention" as JobFilter, tone: "attention" as const }
    ],
    [attentionJobs, queuedJobs, runningJobs]
  );
  const createGuidance = useMemo(
    () => buildCreateGuidance(formState.model, formState.source_type, files.length),
    [files.length, formState.model, formState.source_type]
  );
  const pendingImageCount = useMemo(() => files.filter((file) => isImageLikeFile(file)).length, [files]);
  const pendingVideoCount = useMemo(() => files.filter((file) => isVideoLikeFile(file)).length, [files]);
  const pendingUnknownCount = files.length - pendingImageCount - pendingVideoCount;
  const pendingTotalSize = useMemo(() => files.reduce((total, file) => total + file.size, 0), [files]);
  const pendingTypeSummary = useMemo(
    () =>
      [
        pendingImageCount > 0 ? `图片 ${pendingImageCount}` : null,
        pendingVideoCount > 0 ? `视频 ${pendingVideoCount}` : null,
        pendingUnknownCount > 0 ? `其他 ${pendingUnknownCount}` : null
      ]
        .filter(Boolean)
        .join(" / ") || "暂无",
    [pendingImageCount, pendingUnknownCount, pendingVideoCount]
  );
  const createReadiness = useMemo(
    () => buildCreateReadiness(serviceReady, formState.model, formState.source_type, files.length, pendingImageCount, pendingVideoCount),
    [files.length, formState.model, formState.source_type, pendingImageCount, pendingVideoCount, serviceReady]
  );
  const summary = bootstrap?.summary ?? {
    total: jobs.length,
    running: jobs.filter((item) => item.job.status === "running").length,
    finished: jobs.filter((item) => item.job.status === "finished").length,
    failed: jobs.filter((item) => item.job.status === "failed").length,
    cancelled: jobs.filter((item) => item.job.status === "cancelled").length
  };
  const workspaceTabMeta = useMemo(
    () =>
      workspaceTabs.map((tab) => ({
        ...tab,
        count:
          tab.key === "jobs"
            ? summary.total
            : tab.key === "advisor"
              ? advisorCandidateCount
              : tab.key === "overview"
                ? summary.running
                : undefined
      })),
    [advisorCandidateCount, summary.running, summary.total]
  );
  const runningSelectedJob = selectedJob?.job.status === "running";
  const selectedJobPollMs = runningSelectedJob ? 4000 : 15000;
  const canDispatchSelectedJob = selectedJob
    ? canDispatchJobStatus(selectedJob.job.status)
    : false;

  useEffect(() => {
    let cancelled = false;

    async function boot() {
      await loadDesktopBackendStatus();

      for (let attempt = 1; attempt <= 35; attempt += 1) {
        if (cancelled) {
          return;
        }

        try {
          await refreshBootstrap();
          setServiceState("ready");
          setServiceMessage("本地服务已就绪");
          await loadJobs(false);
          await loadSamples(false);
          return;
        } catch {
          setServiceState("starting");
          setServiceMessage(`正在启动本地服务... ${attempt}/35`);
          await delay(650);
        }
      }

      if (!cancelled) {
        setServiceState("degraded");
        setServiceMessage("本地服务没有按时响应。你仍然可以查看页面，任务操作会在服务恢复后可用。");
      }
    }

    void boot();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => void loadDesktopBackendStatus(), serviceReady ? 8000 : 1200);
    return () => window.clearInterval(timer);
  }, [serviceReady]);

  useEffect(() => {
    if (!previewAsset) {
      return;
    }

    function handleEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setPreviewAsset(null);
      }
    }

    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, [previewAsset]);

  useEffect(() => {
    if (!serviceReady) {
      return;
    }
    const intervalMs = hasRunningJobs ? 4000 : 12000;
    const timer = window.setInterval(() => void loadJobs(false), intervalMs);
    return () => window.clearInterval(timer);
  }, [hasRunningJobs, serviceReady]);

  useEffect(() => {
    if (!serviceReady) {
      return;
    }
    void loadSamples(false);
    const timer = window.setInterval(() => void loadSamples(false), 60000);
    return () => window.clearInterval(timer);
  }, [serviceReady]);

  useEffect(() => {
    if (!selectedJobId && jobs[0]) {
      setSelectedJobId(jobs[0].job.job_id);
    }
  }, [jobs, selectedJobId]);

  useEffect(() => {
    if (!serviceReady || !selectedJobId) {
      setSelectedJob(null);
      return;
    }
    void loadJobDetail(selectedJobId, false);
    const timer = window.setInterval(() => void loadJobDetail(selectedJobId, false), selectedJobPollMs);
    return () => window.clearInterval(timer);
  }, [selectedJobId, selectedJobPollMs, serviceReady]);

  useEffect(() => {
    if (!serviceReady || activeWorkspace !== "system" || deploymentStatus) {
      return;
    }
    void loadDeploymentStatus(false, false);
  }, [activeWorkspace, deploymentStatus, serviceReady]);

  async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
    let response: Response;
    try {
      response = await fetch(`${API_BASE}${path}`, init);
    } catch {
      throw new Error("本地服务暂时不可用，请稍等几秒。");
    }

    if (!response.ok) {
      let message = `请求失败：${response.status}`;
      try {
        const payload = (await response.json()) as { detail?: string };
        if (payload.detail) {
          message = payload.detail;
        }
      } catch {
        const text = await response.text();
        if (text) {
          message = text;
        }
      }
      throw new Error(message);
    }
    return (await response.json()) as T;
  }

  function assetUrl(path: string) {
    if (/^(https?:|data:|blob:)/.test(path)) {
      return path;
    }
    return `${API_BASE}${path.startsWith("/") ? "" : "/"}${path}`;
  }

  async function refreshBootstrap() {
    const payload = await fetchJson<BootstrapPayload>("/api/bootstrap");
    setBootstrap(payload);
    return payload;
  }

  async function loadDesktopBackendStatus() {
    try {
      const status = await invoke<BackendStatusPayload>("backend_status");
      setBackendStatus(status);
      if (!status.running) {
        setServiceState((current) => (current === "starting" ? current : "degraded"));
        setServiceMessage(status.message || "本地服务暂时不可用。");
      }
    } catch {
      setBackendStatus(null);
    }
  }

  async function recoverBackend(mode: "ensure" | "restart", announce = false) {
    if (recoveryInFlightRef.current) {
      return;
    }
    recoveryInFlightRef.current = true;
    setRecoveringService(true);
    setServiceState("starting");
    setServiceMessage(mode === "restart" ? "正在重启本地服务..." : "正在尝试恢复本地服务...");

    try {
      const command = mode === "restart" ? "restart_backend" : "ensure_backend_now";
      const status = await invoke<BackendStatusPayload>(command);
      setBackendStatus(status);

      if (!status.running) {
        setServiceState("degraded");
        setServiceMessage(status.message || "本地服务恢复失败。");
        if (announce) {
          setErrorMessage(status.message || "本地服务恢复失败。");
        }
        return;
      }

      await refreshBootstrap();
      setServiceState("ready");
      setServiceMessage("本地服务已恢复并重新连通。");
      setErrorMessage(null);
      if (announce) {
        setInfoMessage(mode === "restart" ? "本地服务已重启。" : "本地服务已恢复。");
      }
      await loadJobs(false);
      await loadSamples(false);
      if (selectedJobId) {
        await loadJobDetail(selectedJobId, false);
      }
    } catch (error) {
      const message = friendlyError(error, "本地服务恢复失败。");
      setServiceState("degraded");
      setServiceMessage(message);
      if (announce) {
        setErrorMessage(message);
      }
    } finally {
      recoveryInFlightRef.current = false;
      setRecoveringService(false);
    }
  }

  function handleServiceFailure(error: unknown, fallbackMessage: string, showError: boolean) {
    const message = friendlyError(error, fallbackMessage);
    setServiceState("degraded");
    setServiceMessage("本地服务连接中断，正在尝试自动恢复...");
    setBackendStatus((current) =>
      current
        ? { ...current, running: false, message }
        : {
            running: false,
            managed_by_tauri: false,
            message,
            backend_root: null,
            log_path: null
          }
    );
    if (showError) {
      setErrorMessage(message);
    }
    void recoverBackend("ensure", false);
  }

  async function loadJobs(showError = true) {
    try {
      const payload = await fetchJson<JobsListPayload>("/api/jobs");
      setJobs(payload.jobs);
      setBootstrap((current) => (current ? { ...current, summary: payload.summary } : current));
      if (serviceState !== "ready") {
        setServiceState("ready");
        setServiceMessage("本地服务已就绪");
      }
    } catch (error) {
      handleServiceFailure(error, "加载任务列表失败。", showError);
    }
  }

  async function loadSamples(showError = true) {
    try {
      const payload = await fetchJson<SamplesPayload>("/api/samples");
      setSamplesPayload(payload);
      setSamplesError(null);
    } catch (error) {
      const message = friendlyError(error, "样例库接口暂不可用。");
      setSamplesError(
        /404|not found/i.test(message)
          ? "样例库接口暂未上线；后端提供 /api/samples 后这里会自动刷新。"
          : message
      );
      if (showError) {
        setErrorMessage(message);
      }
    }
  }

  async function loadDeploymentStatus(showError = true, refresh = false) {
    setDeploymentLoading(true);
    try {
      const payload = await fetchJson<DeploymentStatusPayload>(`/api/deployment/status${refresh ? "?refresh=true" : ""}`);
      setDeploymentStatus(payload);
      setDeploymentError(null);
    } catch (error) {
      const message = friendlyError(error, "远端部署状态读取失败。");
      setDeploymentError(message);
      if (showError) {
        setErrorMessage(message);
      }
    } finally {
      setDeploymentLoading(false);
    }
  }

  async function openAdvisorSettings() {
    setAdvisorConfigLoading(true);
    setErrorMessage(null);
    try {
      const payload = await fetchJson<AdvisorConfig>("/api/advisor/config");
      setAdvisorForm({
        enabled: payload.enabled,
        base_url: payload.base_url,
        api_key: "",
        model: payload.model || "gpt-4o-mini",
        has_api_key: Boolean(payload.has_api_key),
      });
      setAdvisorModalOpen(true);
    } catch (error) {
      setErrorMessage(friendlyError(error, "读取 AI 配置失败。"));
    } finally {
      setAdvisorConfigLoading(false);
    }
  }

  async function saveAdvisorSettings(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setAdvisorConfigSaving(true);
    setErrorMessage(null);
    setInfoMessage(null);
    try {
      const payload = await fetchJson<AdvisorConfig>("/api/advisor/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          enabled: advisorForm.enabled,
          base_url: advisorForm.base_url,
          api_key: advisorForm.api_key,
          model: advisorForm.model,
        }),
      });
      setBootstrap((current) => (current ? { ...current, advisor: payload } : current));
      setAdvisorModalOpen(false);
      setAdvisorForm((current) => ({ ...current, api_key: "", has_api_key: Boolean(payload.has_api_key) }));
      setInfoMessage(payload.configured ? "AI 评估配置已保存并可用。" : "AI 配置已保存，但还未达到可用状态。");
    } catch (error) {
      setErrorMessage(friendlyError(error, "保存 AI 配置失败。"));
    } finally {
      setAdvisorConfigSaving(false);
    }
  }

  async function loadJobDetail(jobId: string, showError = true) {
    try {
      setSelectedJob(await fetchJson<JobPayload>(`/api/jobs/${jobId}`));
      if (serviceState !== "ready") {
        setServiceState("ready");
        setServiceMessage("本地服务已就绪");
      }
    } catch (error) {
      handleServiceFailure(error, "加载任务详情失败。", showError);
    }
  }

  async function saveJobEvaluation(jobId: string, payload: EvaluationPayload) {
    setSavingEvaluation(true);
    setErrorMessage(null);
    setInfoMessage(null);
    try {
      const updated = await fetchJson<JobPayload & { ok: boolean; evaluation: EvaluationPayload }>(`/api/jobs/${jobId}/evaluation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      setSelectedJob(updated);
      setInfoMessage(`任务 ${jobId} 的人工评分已保存。`);
      await loadJobs(false);
    } catch (error) {
      setErrorMessage(friendlyError(error, "保存人工评分失败。"));
    } finally {
      setSavingEvaluation(false);
    }
  }

  function updateFormField(key: keyof FormState, value: string) {
    setFormState((current) => ({ ...current, [key]: value }));
    if (isParamFieldKey(key)) {
      setActivePresets((current) => ({ ...current, [formState.model as PresetModel]: null }));
    }
  }

  function updateModel(value: string) {
    setFormState((current) => {
      if (value === "monst3r") {
        return {
          ...current,
          ...defaultMonst3rParams,
          model: value,
          source_type: current.source_type === "images" ? "frames" : current.source_type
        };
      }
      return {
        ...current,
        ...defaultDust3rParams,
        model: value,
        source_type: "images"
      };
    });
    setActivePresets((current) => ({
      ...current,
      [value as PresetModel]: "standard"
    }));
  }

  function openWorkspace(tab: WorkspaceTab, jobId?: string) {
    if (jobId) {
      setSelectedJobId(jobId);
    }
    setActiveWorkspace(tab);
  }

  function focusJobLane(filter: JobFilter, laneJobs: JobListItem[]) {
    setJobQuery("");
    setJobFilter(filter);
    if (laneJobs[0]) {
      setSelectedJobId(laneJobs[0].job.job_id);
    }
  }

  function moveSelectedJob(offset: -1 | 1) {
    if (filteredJobs.length === 0) {
      return;
    }
    const baseIndex = selectedFilteredIndex >= 0 ? selectedFilteredIndex : 0;
    const nextIndex = Math.max(0, Math.min(filteredJobs.length - 1, baseIndex + offset));
    setSelectedJobId(filteredJobs[nextIndex].job.job_id);
  }

  async function copyText(value: string, label: string) {
    try {
      await navigator.clipboard.writeText(value);
      setInfoMessage(`${label} 已复制。`);
      setErrorMessage(null);
    } catch {
      setErrorMessage(`复制${label}失败，请稍后重试。`);
    }
  }

  function applyPreset(preset: PresetKey) {
    setFormState((current) => {
      if (current.model === "monst3r") {
        return {
          ...current,
          ...buildMonst3rPreset(preset)
        };
      }
      return {
        ...current,
        ...buildDust3rPreset(preset, files.length)
      };
    });
    setActivePresets((current) => ({
      ...current,
      [formState.model as PresetModel]: preset
    }));
  }

  function removePendingFile(targetName: string, targetSize: number) {
    setFiles((current) =>
      current.filter((item) => !(item.name === targetName && item.size === targetSize))
    );
  }

  async function handleCreateJob(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setErrorMessage(null);
    setInfoMessage(null);

    if (!serviceReady) {
      setErrorMessage("本地服务还没准备好，请等顶部状态变成“就绪”。");
      return;
    }

    const validationError = validateFiles();
    if (validationError) {
      setErrorMessage(validationError);
      return;
    }

    setSubmitting(true);
    const formData = new FormData();
    formData.append("model", formState.model);
    formData.append("source_type", formState.source_type);
    formData.append("notes", formState.notes);

    const paramDefaults = isImageCollectionModel ? defaultDust3rParams : defaultMonst3rParams;
    Object.keys(paramDefaults).forEach((key) => {
      formData.append(key, formState[key as keyof FormState]);
    });
    files.forEach((file) => formData.append("files", file, file.name));

    try {
      const payload = await fetchJson<JobPayload>("/api/jobs", { method: "POST", body: formData });
      setFiles([]);
      setSelectedJobId(payload.job.job_id);
      setSelectedJob(payload);
      setActiveWorkspace("jobs");
      setInfoMessage(`任务 ${payload.job.job_id} 已创建。`);
      await loadJobs(false);
    } catch (error) {
      setErrorMessage(friendlyError(error, "创建任务失败。"));
    } finally {
      setSubmitting(false);
    }
  }

  function validateFiles() {
    if (files.length === 0) {
      return "请先选择输入文件。";
    }
    if (isImageCollectionModel && files.length < 2) {
      return `${statusModelLabel(formState.model)} 至少需要两张图片。`;
    }
    if (isMonst3r && formState.source_type === "video" && files.length !== 1) {
      return "MonST3R 视频模式请上传 1 个视频文件。";
    }
    if (isMonst3r && formState.source_type !== "video" && files.length < 2) {
      return "MonST3R 帧序列模式至少上传 2 张图片。";
    }
    return null;
  }

  async function postJobAction(path: string, key: string) {
    setActionKey(key);
    setErrorMessage(null);
    setInfoMessage(null);
    try {
      const payload = await fetchJson<JobPayload>(path, { method: "POST" });
      setSelectedJob(payload);
      setSelectedJobId(payload.job.job_id);
      if (key === "advisor") {
        setActiveWorkspace("advisor");
      }
      setInfoMessage(buildActionMessage(key, payload.job.job_id));
      await loadJobs(false);
    } catch (error) {
      setErrorMessage(friendlyError(error, "执行任务操作失败。"));
    } finally {
      setActionKey(null);
    }
  }

  async function openOutput(relativePath: string) {
    if (!selectedJob) {
      return;
    }

    setActionKey(`open:${relativePath}`);
    setErrorMessage(null);
    setInfoMessage(null);

    const formData = new FormData();
    formData.append("relative_path", relativePath);

    try {
      const payload = await fetchJson<{ ok: boolean; path: string }>(
        `/api/jobs/${selectedJob.job.job_id}/open-output`,
        { method: "POST", body: formData }
      );
      setInfoMessage(`已尝试用本地默认程序打开：${payload.path}`);
    } catch (error) {
      setErrorMessage(friendlyError(error, "打开本地产物失败。"));
    } finally {
      setActionKey(null);
    }
  }

  function openPreviewAsset(asset: PreviewAsset) {
    setPreviewAsset(asset);
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand-block">
          <span className="brand-mark">K</span>
          <div>
            <h1>KYKT Vision</h1>
            <p>本地选择输入，服务器运行模型，结果自动回传。</p>
          </div>
        </div>
        <div className="header-actions">
          <StatusBadge state={serviceState} label={serviceStatusLabel(serviceState)} />
          <button
            className="ghost-button"
            onClick={() => void recoverBackend("restart", true)}
            disabled={recoveringService}
          >
            {recoveringService ? "恢复中..." : "重启本地服务"}
          </button>
          <button className="ghost-button" onClick={() => void loadJobs(true)} disabled={!serviceReady}>
            刷新
          </button>
        </div>
      </header>

      <main className="workspace">
        <nav className="workspace-menu" aria-label="主菜单">
          {workspaceTabMeta.map((tab) => (
            <button
              key={tab.key}
              className={`workspace-menu-item ${activeWorkspace === tab.key ? "active" : ""}`}
              onClick={() => openWorkspace(tab.key)}
              type="button"
            >
              <span className="workspace-menu-label">{tab.label}</span>
              <small>{tab.note}</small>
              {typeof tab.count === "number" ? <span className="workspace-menu-count">{tab.count}</span> : null}
            </button>
          ))}
        </nav>

        <div className="status-strip">
          <section className={`service-card ${serviceState}`}>
            <div>
              <span className="mini-label">本地服务</span>
              <strong>{serviceMessage}</strong>
            </div>
            <div className="service-card-copy">
              <p>{backendStatusText(backendStatus)}</p>
              <div className="service-card-actions">
                <button
                  className="ghost-button small"
                  onClick={() => void recoverBackend("ensure", true)}
                  disabled={recoveringService}
                  type="button"
                >
                  立即探测
                </button>
                <button
                  className="ghost-button small"
                  onClick={() => void recoverBackend("restart", true)}
                  disabled={recoveringService}
                  type="button"
                >
                  强制重启
                </button>
              </div>
            </div>
          </section>

          <section className={`service-card advisor-card ${advisorReady ? "ready" : advisorState.enabled ? "starting" : "degraded"}`}>
            <div>
              <span className="mini-label">AI 评估</span>
              <strong>
                {advisorReady
                  ? `已就绪 · ${advisorState.model || "已配置模型"}`
                  : advisorState.enabled
                    ? "已启用但尚未完整配置"
                    : "尚未启用"}
              </strong>
            </div>
            <div className="service-card-copy">
              <p>{advisorReady ? "适合在任务完成、失败或准备汇报时调用。" : advisorState.message}</p>
              <div className="service-card-actions">
                <button className="ghost-button small" onClick={() => openWorkspace("advisor")} type="button">
                  打开 AI 工作台
                </button>
                <button className="ghost-button small" onClick={() => void openAdvisorSettings()} disabled={advisorConfigLoading} type="button">
                  {advisorConfigLoading ? "读取中..." : "配置 AI"}
                </button>
                {advisorReady && activeJob && isAdvisorSuggested(activeJob.job.status) ? (
                  <button
                    className="ghost-button small"
                    onClick={() => void postJobAction(`/api/jobs/${activeJob.job.job_id}/advisor/evaluate`, "advisor")}
                    disabled={actionKey === "advisor"}
                    type="button"
                  >
                    评估当前任务
                  </button>
                ) : null}
              </div>
            </div>
          </section>
        </div>

        {infoMessage ? <MessageBanner kind="info" message={infoMessage} /> : null}
        {errorMessage ? <MessageBanner kind="error" message={errorMessage} /> : null}

        {activeWorkspace === "overview" ? (
          <OverviewCommandCenter
            focusJob={focusJob}
            summary={summary}
            runningJobs={runningJobs}
            attentionJobs={attentionJobs}
            activeModelCatalog={activeModelCatalog}
            deferredModelCatalog={deferredModelCatalog}
            samplesPayload={samplesPayload}
            samplesError={samplesError}
            modelCatalog={modelCatalog}
            openWorkspace={openWorkspace}
            activeJob={activeJob}
            advisorState={advisorState}
            actionKey={actionKey}
            postJobAction={postJobAction}
            openAdvisorSettings={openAdvisorSettings}
            copyText={copyText}
          />
        ) : null}

        {activeWorkspace === "create" ? (
        <section className="layout-grid create-layout">
          <article className="panel create-panel">
            <PanelTitle eyebrow="新建任务" title="选择模型和输入" />
            <div className="create-run-strip" aria-label="创建任务检查">
              {createReadiness.map((item) => (
                <article className={`create-run-item ${item.tone}`} key={item.label}>
                  <span>{item.label}</span>
                  <strong>{item.value}</strong>
                </article>
              ))}
            </div>
            <form className="form-stack" onSubmit={handleCreateJob}>
              <section className="create-workbench-grid">
                <article className="create-block create-config-block">
                  <div className="create-block-head">
                    <div>
                      <span className="mini-label">任务配置</span>
                      <strong>模型与输入来源</strong>
                    </div>
                    <span className="section-pill">{selectedModel?.label ?? "未选择模型"}</span>
                  </div>

                  <div className="form-row">
                    <label className="field">
                      <span>模型</span>
                      <select value={formState.model} onChange={(event) => updateModel(event.target.value)}>
                        {bootstrapData.models.map((item) => (
                          <option key={item.value} value={item.value}>
                            {item.label}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className="field">
                      <span>输入类型</span>
                      <select
                        value={formState.source_type}
                        onChange={(event) => updateFormField("source_type", event.target.value)}
                      >
                        {bootstrapData.source_types
                          .filter((item) => allowedSourceTypesForModel(formState.model).includes(item.value))
                          .map((item) => (
                            <option key={item.value} value={item.value}>
                              {item.label}
                            </option>
                          ))}
                      </select>
                    </label>
                  </div>

                  <label className="field">
                    <span>备注</span>
                    <textarea
                      rows={3}
                      value={formState.notes}
                      onChange={(event) => updateFormField("notes", event.target.value)}
                      placeholder="比如：箱子双图测试 / 室内视频 / 想比较 MonST3R 和 DUSt3R"
                    />
                  </label>
                </article>

                <article className="create-block create-staging-block">
                  <div className="create-block-head">
                    <div>
                      <span className="mini-label">输入 Staging</span>
                      <strong>本地待上传文件</strong>
                    </div>
                    <span className="section-pill">{files.length} 个文件</span>
                  </div>

                  <label className="dropzone">
                    <input
                      type="file"
                      multiple
                      onChange={(event) => setFiles(Array.from(event.target.files ?? []))}
                    />
                    <span>点击选择文件</span>
                    <small>{inputHint(formState.model, formState.source_type)}</small>
                  </label>

                  <div className="staging-stats">
                    <SummaryStat label="文件数量" value={String(files.length)} />
                    <SummaryStat label="总大小" value={formatFileSize(pendingTotalSize)} />
                    <SummaryStat label="类型分布" value={pendingTypeSummary} />
                  </div>

                  {files.length > 0 ? (
                    <div className="staging-table" role="table" aria-label="待上传文件矩阵">
                      <div className="staging-table-head" role="row">
                        <span role="columnheader">文件名</span>
                        <span role="columnheader">类型</span>
                        <span role="columnheader">大小</span>
                        <span role="columnheader">操作</span>
                      </div>
                      <div className="staging-table-body">
                        {files.map((file) => (
                          <div className="staging-row" key={`${file.name}-${file.size}`} role="row">
                            <span className="staging-name" role="cell" title={file.name}>
                              {file.name}
                            </span>
                            <span role="cell">{pendingFileRoleLabel(file)}</span>
                            <span role="cell">{formatFileSize(file.size)}</span>
                            <button
                              className="ghost-button small staging-remove-button"
                              type="button"
                              onClick={() => removePendingFile(file.name, file.size)}
                            >
                              移除
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="empty-state">先拖入或选择文件，staging 区会显示完整文件矩阵。</div>
                  )}
                </article>

                <article className="create-block create-guidance-block">
                  <div className="create-guidance">
                    <div className="guide-head">
                      <div>
                        <strong>当前建议</strong>
                        <p>{selectedModel?.description ?? "根据模型类型自动给出最稳妥的起步建议。"}</p>
                      </div>
                      <span className="section-pill">{files.length} 个待上传</span>
                    </div>
                    <ul className="guide-list">
                      {createGuidance.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>
                </article>

                <article className="create-block create-params-block">
                  <details className="advanced-panel" open>
                    <summary>高级参数</summary>
                    <div className="advanced-panel-intro">
                      这些参数已经整理成推荐档位了。直接优先选择带“推荐”字样的选项，只有做对比实验时再切到其它档。
                    </div>
                    <div className="preset-strip">
                      <div className="preset-strip-head">
                        <strong>一键预设</strong>
                        <span>
                          {activePresets[formState.model as PresetModel]
                            ? `当前：${presetLabel(activePresets[formState.model as PresetModel])}`
                            : "当前：已手动调整"}
                        </span>
                      </div>
                      <div className="preset-pills">
                        {presetDescriptors.map((preset) => (
                          <button
                            key={preset.key}
                            className={`preset-pill ${
                              activePresets[formState.model as PresetModel] === preset.key ? "active" : ""
                            }`}
                            type="button"
                            onClick={() => applyPreset(preset.key)}
                          >
                            <strong>{preset.label}</strong>
                            <span>{preset.note}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                    {isImageCollectionModel ? (
                      <div className="param-grid">
                        {Object.keys(defaultDust3rParams).map((key) => (
                          <ParamField
                            key={key}
                            name={key}
                            value={formState[key as keyof FormState]}
                            choices={dust3rParamChoices[key as keyof typeof defaultDust3rParams]}
                            onChange={(value) => updateFormField(key as keyof FormState, value)}
                          />
                        ))}
                      </div>
                    ) : (
                      <Monst3rParams formState={formState} updateFormField={updateFormField} />
                    )}
                  </details>
                </article>
              </section>

              <div className="create-submit-dock">
                <div>
                  <span className="mini-label">Launch</span>
                  <strong>{buildCreateLaunchHeadline(serviceReady, files.length)}</strong>
                  <p>{buildCreateLaunchMessage(serviceReady, formState.model, formState.source_type, files.length)}</p>
                </div>
                <button className="primary-button" disabled={!serviceReady || submitting} type="submit">
                  {submitting ? "创建中..." : `创建 ${selectedModel?.label ?? "模型"} 任务`}
                </button>
              </div>
            </form>
          </article>

          <aside className="panel create-support-panel">
            <PanelTitle eyebrow="输入规范" title="先把原料准备对" />
            <div className="support-checklist">
              {buildCaptureChecklist(formState.model, formState.source_type, files.length).map((item) => (
                <article className="support-check-item" key={item.title}>
                  <strong>{item.title}</strong>
                  <p>{item.body}</p>
                </article>
              ))}
            </div>
          </aside>
        </section>
        ) : null}

        {activeWorkspace === "jobs" ? (
          <section className="layout-grid jobs-layout">
            <aside className="panel side-panel">
              <PanelTitle eyebrow="任务列表" title={`${summary.total} 个任务`} />
              <div className="mini-stats">
                <MiniStat label="总任务" value={summary.total} />
                <MiniStat label="运行" value={summary.running} />
                <MiniStat label="待处理" value={attentionJobs.length} />
                <MiniStat label="完成" value={summary.finished} />
              </div>
              <div className="jobs-side-tools">
                <label className="field compact job-search-field">
                  <span>快速检索</span>
                  <input
                    value={jobQuery}
                    onChange={(event) => setJobQuery(event.target.value)}
                    placeholder="任务ID / 模型 / 状态 / 输入类型"
                  />
                </label>
                <div className="job-lane-grid" aria-label="任务队列入口">
                  {jobLaneCards.map((lane) => (
                    <article className={`job-lane-card ${lane.tone}`} key={lane.key}>
                      <div className="job-lane-head">
                        <span>{lane.label}</span>
                        <strong>{lane.jobs.length}</strong>
                      </div>
                      <p>{lane.jobs[0] ? `${lane.jobs[0].job.job_id} · ${lane.jobs[0].phase_display.label}` : "暂无可定位任务"}</p>
                      <button className="ghost-button small" type="button" onClick={() => focusJobLane(lane.filter, lane.jobs)} disabled={lane.jobs.length === 0}>
                        定位首条
                      </button>
                    </article>
                  ))}
                </div>
              </div>
              <div className="job-toolbar">
                <div className="filter-pills">
                  <FilterPill
                    active={jobFilter === "all"}
                    count={summary.total}
                    label="全部"
                    onClick={() => setJobFilter("all")}
                  />
                  <FilterPill
                    active={jobFilter === "running"}
                    count={runningJobs.length}
                    label="运行中"
                    onClick={() => setJobFilter("running")}
                  />
                  <FilterPill
                    active={jobFilter === "attention"}
                    count={attentionJobs.length}
                    label="待处理"
                    onClick={() => setJobFilter("attention")}
                  />
                  <FilterPill
                    active={jobFilter === "finished"}
                    count={finishedJobs.length}
                    label="已完成"
                    onClick={() => setJobFilter("finished")}
                  />
                </div>
                <div className="jobs-selection-strip">
                  <div className="jobs-selection-copy">
                    <span className="mini-label">筛选上下文</span>
                    <strong>
                      {selectedFilteredIndex >= 0
                        ? `第 ${selectedFilteredIndex + 1} / ${filteredJobs.length} 条`
                        : filteredJobs.length > 0
                          ? "当前选中项不在筛选结果中"
                          : "当前筛选为空"}
                    </strong>
                    <p>
                      {selectedListJob
                        ? `${statusModelLabel(selectedListJob.job.model)} · ${statusLabel(selectedListJob.job.status)}`
                        : "先在列表里选中一条任务。"}
                      {selectedFilteredModelCount > 0 ? ` · 同模型 ${selectedFilteredModelCount} 条` : ""}
                    </p>
                  </div>
                  <div className="jobs-selection-actions">
                    <button
                      className="ghost-button small"
                      type="button"
                      onClick={() => moveSelectedJob(-1)}
                      disabled={filteredJobs.length === 0 || selectedFilteredIndex <= 0}
                    >
                      上一条
                    </button>
                    <button
                      className="ghost-button small"
                      type="button"
                      onClick={() => moveSelectedJob(1)}
                      disabled={filteredJobs.length === 0 || selectedFilteredIndex < 0 || selectedFilteredIndex >= filteredJobs.length - 1}
                    >
                      下一条
                    </button>
                    <button
                      className="ghost-button small"
                      type="button"
                      onClick={() => selectedListJob && void copyText(selectedListJob.job.job_id, "任务ID")}
                      disabled={!selectedListJob}
                    >
                      复制ID
                    </button>
                    <button
                      className="ghost-button small"
                      type="button"
                      onClick={() => void copyText(filteredJobIds.join("\n"), "筛选任务ID")}
                      disabled={filteredJobIds.length === 0}
                    >
                      复制筛选ID
                    </button>
                  </div>
                </div>
                {selectedListJob ? (
                  <div className="jobs-inline-actions">
                    <button
                      className="ghost-button small"
                      type="button"
                      disabled={!canDispatchListJob || actionKey === "dispatch"}
                      onClick={() => void postJobAction(`/api/jobs/${selectedListJob.job.job_id}/dispatch`, "dispatch")}
                    >
                      {actionKey === "dispatch" ? "运行中..." : "运行"}
                    </button>
                    <button
                      className="ghost-button small"
                      type="button"
                      disabled={!canRetryListJob || actionKey === "retry"}
                      onClick={() => void postJobAction(`/api/jobs/${selectedListJob.job.job_id}/retry`, "retry")}
                    >
                      {actionKey === "retry" ? "重试中..." : "重试"}
                    </button>
                    <button
                      className="ghost-button small danger"
                      type="button"
                      disabled={!canCancelListJob || actionKey === "cancel"}
                      onClick={() => void postJobAction(`/api/jobs/${selectedListJob.job.job_id}/cancel`, "cancel")}
                    >
                      {actionKey === "cancel" ? "取消中..." : "取消"}
                    </button>
                  </div>
                ) : null}
              </div>
              <div className="job-list">
                {filteredJobs.length > 0 ? (
                  filteredJobs.map((item) => (
                    <button
                      key={item.job.job_id}
                      className={`job-card ${selectedJobId === item.job.job_id ? "active" : ""} ${item.job.status}`}
                      onClick={() => setSelectedJobId(item.job.job_id)}
                      type="button"
                    >
                      <div className="job-card-head">
                        <strong>{item.job.job_id}</strong>
                        <span className="job-card-percent">{item.phase_display.percent}%</span>
                      </div>
                      <div>
                        <p>{item.phase_display.label}</p>
                        <span className="job-card-meta">
                          {statusModelLabel(item.job.model)} · {statusLabel(item.job.status)}
                        </span>
                        <span className="job-card-stage">{currentStepLabel(item.phase_display.steps)}</span>
                        <p className="job-card-message">
                          {item.job.progress_message || item.phase_display.description}
                        </p>
                      </div>
                      <div className="progress-track">
                        <div className="progress-fill" style={{ width: `${item.phase_display.percent}%` }} />
                      </div>
                    </button>
                  ))
                ) : (
                  <div className="empty-state">
                    {normalizedJobQuery
                      ? `没有匹配“${jobQuery.trim()}”的任务。`
                      : jobFilter === "all"
                      ? "暂无任务。先在“文件与新建”里选择输入文件，创建第一条任务。"
                      : `当前筛选下没有${jobFilterLabel(jobFilter)}任务。`}
                  </div>
                )}
              </div>
            </aside>

            <section className="panel detail-panel">
              <PanelTitle eyebrow="任务详情" title={selectedJob?.job.job_id ?? "尚未选择任务"} />
              {selectedJob ? (
                <JobDetail
                  selectedJob={selectedJob}
                  advisorState={advisorState}
                  actionKey={actionKey}
                  savingEvaluation={savingEvaluation}
                  canDispatch={canDispatchSelectedJob}
                  running={Boolean(runningSelectedJob)}
                  assetUrl={assetUrl}
                  onAction={postJobAction}
                  onSaveEvaluation={saveJobEvaluation}
                  onConfigureAdvisor={() => void openAdvisorSettings()}
                  onOpenOutput={openOutput}
                  onPreviewAsset={openPreviewAsset}
                  onCopy={copyText}
                />
              ) : (
                <div className="empty-state large">
                  这里会显示任务进度、输入预览、输出结果和日志。现在这些内容只会在“运行与结果”里集中展示，不再塞满首页。
                </div>
              )}
            </section>
          </section>
        ) : null}

        {activeWorkspace === "advisor" ? (
          <section className="advisor-layout">
            <article className="panel advisor-main-panel">
              <PanelTitle eyebrow="AI 工作台" title={selectedJob?.job.job_id ?? "先从任务中心选中一条任务"} />
              <AdvisorWorkbench
                job={selectedJob}
                advisorState={advisorState}
                actionKey={actionKey}
                onEvaluate={(jobId) => void postJobAction(`/api/jobs/${jobId}/advisor/evaluate`, "advisor")}
                onConfigure={() => void openAdvisorSettings()}
                onCopy={copyText}
              />
            </article>

            <aside className="panel advisor-side-panel">
              <PanelTitle eyebrow="什么时候用 AI" title="别让它孤零零地藏在按钮里" />
              <div className="support-checklist">
                {buildAdvisorChecklist(advisorReady).map((item) => (
                  <article className="support-check-item" key={item.title}>
                    <strong>{item.title}</strong>
                    <p>{item.body}</p>
                  </article>
                ))}
              </div>
            </aside>
          </section>
        ) : null}

        {activeWorkspace === "system" ? (
          <SystemWorkbench
            bootstrapData={bootstrapData}
            deploymentStatus={deploymentStatus}
            deploymentError={deploymentError}
            deploymentLoading={deploymentLoading}
            loadDeploymentStatus={loadDeploymentStatus}
            advisorReady={advisorReady}
            advisorState={advisorState}
            advisorConfigLoading={advisorConfigLoading}
            openAdvisorSettings={openAdvisorSettings}
            activeModelCatalog={activeModelCatalog}
            deferredModelCatalog={deferredModelCatalog}
            samplesPayload={samplesPayload}
            samplesError={samplesError}
            modelCatalog={modelCatalog}
            openWorkspace={openWorkspace}
            copyText={copyText}
            serviceMessage={serviceMessage}
            backendStatus={backendStatus}
          />
        ) : null}
      </main>

      {advisorModalOpen ? (
        <div className="settings-modal-backdrop" onClick={() => setAdvisorModalOpen(false)} role="presentation">
          <div className="settings-modal" onClick={(event) => event.stopPropagation()} role="dialog" aria-modal="true">
            <div className="preview-modal-head">
              <div>
                <span className="mini-label">AI 配置</span>
                <strong>填写 OpenAI 兼容接口</strong>
                <p>保存后会立即刷新 AI 状态。已保存的密钥不会明文回显，留空会保持当前密钥不变。</p>
              </div>
              <button className="ghost-button small" onClick={() => setAdvisorModalOpen(false)} type="button">
                关闭
              </button>
            </div>

            <form className="form-stack settings-form" onSubmit={saveAdvisorSettings}>
              <label className="field">
                <span>启用 AI 评估</span>
                <select
                  value={advisorForm.enabled ? "true" : "false"}
                  onChange={(event) =>
                    setAdvisorForm((current) => ({ ...current, enabled: event.target.value === "true" }))
                  }
                >
                  <option value="true">开启</option>
                  <option value="false">关闭</option>
                </select>
              </label>

              <label className="field">
                <span>Base URL</span>
                <input
                  value={advisorForm.base_url}
                  onChange={(event) => setAdvisorForm((current) => ({ ...current, base_url: event.target.value }))}
                  placeholder="例如：http://127.0.0.1:3000/v1"
                />
              </label>

              <label className="field">
                <span>API Key</span>
                <input
                  type="password"
                  value={advisorForm.api_key}
                  onChange={(event) => setAdvisorForm((current) => ({ ...current, api_key: event.target.value }))}
                  placeholder={advisorForm.has_api_key ? "已保存，留空则保持不变" : "输入新的 API Key"}
                />
              </label>

              <label className="field">
                <span>Model</span>
                <input
                  value={advisorForm.model}
                  onChange={(event) => setAdvisorForm((current) => ({ ...current, model: event.target.value }))}
                  placeholder="例如：gpt-4o-mini"
                />
              </label>

              <div className="settings-modal-actions">
                <button className="ghost-button" onClick={() => setAdvisorModalOpen(false)} type="button">
                  取消
                </button>
                <button className="primary-button" disabled={advisorConfigSaving} type="submit">
                  {advisorConfigSaving ? "保存中..." : "保存配置"}
                </button>
              </div>
            </form>
          </div>
        </div>
      ) : null}

      {previewAsset ? (
        <div className="preview-modal-backdrop" onClick={() => setPreviewAsset(null)} role="presentation">
          <div className="preview-modal" onClick={(event) => event.stopPropagation()} role="dialog" aria-modal="true">
            <div className="preview-modal-head">
              <div>
                <span className="mini-label">结果预览</span>
                <strong>{previewAsset.name}</strong>
                {previewAsset.note ? <p>{previewAsset.note}</p> : null}
              </div>
              <button className="ghost-button small" onClick={() => setPreviewAsset(null)} type="button">
                关闭
              </button>
            </div>
            <div className="preview-modal-body">
              {previewAsset.kind === "image" ? (
                <img src={previewAsset.url} alt={previewAsset.name} className="preview-modal-image" />
              ) : (
                <video src={previewAsset.url} className="preview-modal-video" controls autoPlay />
              )}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function OverviewCommandCenter(props: {
  focusJob: JobListItem | null;
  summary: BootstrapPayload["summary"];
  runningJobs: JobListItem[];
  attentionJobs: JobListItem[];
  activeModelCatalog: ModelCatalogItem[];
  deferredModelCatalog: ModelCatalogItem[];
  samplesPayload: SamplesPayload | null;
  samplesError: string | null;
  modelCatalog: ModelCatalogItem[];
  openWorkspace: (workspace: WorkspaceTab, jobId?: string) => void;
  activeJob: JobPayload | null;
  advisorState: AdvisorStatus;
  actionKey: string | null;
  postJobAction: (path: string, key: string) => Promise<void>;
  openAdvisorSettings: () => Promise<void> | void;
  copyText: (value: string, label: string) => Promise<void>;
}) {
  const focusJob = props.focusJob;
  return (
    <>
      <section className="overview-grid workbench-overview-grid">
        <article className="panel overview-hero-panel">
          <PanelTitle eyebrow="Focus" title={focusJob ? focusJob.job.job_id : "准备开始今晚测试"} />
          {focusJob ? (
            <div className={`focus-card ${focusJob.job.status}`}>
              <div className="focus-main">
                <div className="focus-copy">
                  <div className="hero-badges">
                    <StatusBadge state={focusJob.job.status} label={statusLabel(focusJob.job.status)} />
                    <span className="hero-tag">{statusModelLabel(focusJob.job.model)}</span>
                    <span className="hero-tag">{sourceTypeLabel(focusJob.job.source_type)}</span>
                  </div>
                  <h3>{focusJob.phase_display.label}</h3>
                  <p>{focusJob.job.progress_message || focusJob.phase_display.description}</p>
                </div>
                <div className="focus-score">{focusJob.phase_display.percent}%</div>
              </div>
              <div className="progress-track large">
                <div className="progress-fill" style={{ width: `${focusJob.phase_display.percent}%` }} />
              </div>
              <div className="focus-meta">
                <span>{currentStepLabel(focusJob.phase_display.steps)}</span>
                <span>{formatDateTime(focusJob.job.created_at)}</span>
                <button className="ghost-button small" onClick={() => props.openWorkspace("jobs", focusJob.job.job_id)} type="button">
                  进入任务中心
                </button>
              </div>
            </div>
          ) : (
            <div className="empty-state large">还没有任务。建议先从 2 张图片的 DUSt3R / MASt3R 或 1 段短视频的 MonST3R 开始，先把一条样例完整跑通。</div>
          )}
        </article>

        <aside className="panel overview-side-panel">
          <PanelTitle eyebrow="Runtime" title="当前调度状态" />
          <div className="kpi-grid">
            <MiniStat label="总任务" value={props.summary.total} />
            <MiniStat label="运行中" value={props.summary.running} />
            <MiniStat label="待处理" value={props.attentionJobs.length} />
            <MiniStat label="已完成" value={props.summary.finished} />
          </div>
          <div className={`overview-callout ${focusJob?.job.status ?? "neutral"}`}>
            <span className="mini-label">当前建议</span>
            <strong>{buildOverviewHeadline(focusJob, props.runningJobs.length, props.attentionJobs.length)}</strong>
            <p>{buildOverviewMessage(focusJob, props.runningJobs.length, props.attentionJobs.length)}</p>
          </div>
          <div className="overview-ops-dock" aria-label="工作台操作入口">
            <button onClick={() => props.openWorkspace("jobs", focusJob?.job.job_id)} type="button">
              <span>Jobs</span>
              <strong>{props.runningJobs.length}</strong>
              <small>运行中</small>
            </button>
            <button onClick={() => props.openWorkspace("jobs")} type="button">
              <span>Fix</span>
              <strong>{props.attentionJobs.length}</strong>
              <small>待处理</small>
            </button>
            <button onClick={() => props.openWorkspace("create")} type="button">
              <span>New</span>
              <strong>{props.summary.total}</strong>
              <small>任务总数</small>
            </button>
            <button onClick={() => props.openWorkspace("system")} type="button">
              <span>Ops</span>
              <strong>{props.activeModelCatalog.filter((item) => item.runnable).length}</strong>
              <small>可运行模型</small>
            </button>
          </div>
        </aside>
      </section>

      <section className="overview-support-grid workbench-overview-support-grid">
        <article className="panel quick-actions-panel">
          <PanelTitle eyebrow="Workflow" title="按桌面工作流推进" />
          <div className="quick-action-grid">
            <button className="quick-action-card" onClick={() => props.openWorkspace("create")} type="button">
              <strong>新建任务</strong>
              <p>选择模型、输入类型和文件，按推荐参数快速起跑。</p>
            </button>
            <button className="quick-action-card" onClick={() => props.openWorkspace("jobs")} type="button">
              <strong>运行与结果</strong>
              <p>把任务列表、结果、日志和人工/AI 评估集中到一个主工作区。</p>
            </button>
            <button className="quick-action-card" onClick={() => props.openWorkspace("advisor")} type="button">
              <strong>AI 评估</strong>
              <p>在任务完成、失败或准备汇报时生成诊断和下一步建议。</p>
            </button>
            <button className="quick-action-card" onClick={() => props.openWorkspace("system")} type="button">
              <strong>系统与部署</strong>
              <p>检查本地服务、远端 active 3R 部署、阻塞项与缓存状态。</p>
            </button>
          </div>
        </article>

        <article className="panel advisor-overview-panel">
          <PanelTitle eyebrow="Advisor" title="AI 工作台" />
          <AdvisorWorkbench
            job={props.activeJob}
            advisorState={props.advisorState}
            actionKey={props.actionKey}
            onEvaluate={(jobId) => void props.postJobAction(`/api/jobs/${jobId}/advisor/evaluate`, "advisor")}
            onConfigure={() => void props.openAdvisorSettings()}
            onCopy={props.copyText}
            compact
          />
        </article>

        <ModelRoadmapPanel activeModels={props.activeModelCatalog} deferredModels={props.deferredModelCatalog} compact />
        <SampleMatrixPanel
          samplesPayload={props.samplesPayload}
          errorMessage={props.samplesError}
          modelCatalog={props.modelCatalog}
          onLocateJob={(jobId) => props.openWorkspace("jobs", jobId)}
          onCopy={props.copyText}
          compact
        />
      </section>
    </>
  );
}

function SystemWorkbench(props: {
  bootstrapData: BootstrapPayload;
  deploymentStatus: DeploymentStatusPayload | null;
  deploymentError: string | null;
  deploymentLoading: boolean;
  loadDeploymentStatus: (showError?: boolean, refresh?: boolean) => Promise<void>;
  advisorReady: boolean;
  advisorState: AdvisorStatus;
  advisorConfigLoading: boolean;
  openAdvisorSettings: () => Promise<void> | void;
  activeModelCatalog: ModelCatalogItem[];
  deferredModelCatalog: ModelCatalogItem[];
  samplesPayload: SamplesPayload | null;
  samplesError: string | null;
  modelCatalog: ModelCatalogItem[];
  openWorkspace: (workspace: WorkspaceTab, jobId?: string) => void;
  copyText: (value: string, label: string) => Promise<void>;
  serviceMessage: string;
  backendStatus: BackendStatusPayload | null;
}) {
  const deploymentRows = buildDeploymentComponentRows(props.deploymentStatus);

  return (
    <section className="system-grid workbench-system-grid">
      <article className="panel">
        <PanelTitle eyebrow="Local Runtime" title="本地服务状态" />
        <div className="support-checklist compact-stack">
          <article className="support-check-item">
            <strong>服务状态</strong>
            <p>{props.serviceMessage}</p>
          </article>
          <article className="support-check-item">
            <strong>后端说明</strong>
            <p>{backendStatusText(props.backendStatus)}</p>
          </article>
        </div>
      </article>

      <article className="panel">
        <PanelTitle eyebrow="Remote Target" title="服务器连接信息" />
        <div className="support-checklist compact-stack">
          <article className="support-check-item">
            <strong>SSH 目标</strong>
            <p>{props.bootstrapData.server.user}@{props.bootstrapData.server.host}:{props.bootstrapData.server.port}</p>
          </article>
          <article className="support-check-item">
            <strong>远端根目录</strong>
            <p>{props.bootstrapData.server.remote_root}</p>
          </article>
        </div>
      </article>

      <article className="panel deployment-console-panel">
        <PanelTitle eyebrow="Deployment" title="Active 3R 部署控制台" />
        {deploymentRows.length > 0 ? (
          <div className="deployment-readiness-table" role="table" aria-label="Active 3R 部署 readiness">
            <div className="deployment-readiness-head" role="row">
              <span role="columnheader">模型</span>
              <span role="columnheader">目录</span>
              <span role="columnheader">环境</span>
              <span role="columnheader">文件</span>
              <span role="columnheader">权重</span>
            </div>
            {deploymentRows.map((row) => (
              <div className={`deployment-readiness-row ${row.tone}`} key={row.component} role="row">
                <strong role="cell">{modelDisplayName(row.component, props.modelCatalog)}</strong>
                <span role="cell">{row.directory}</span>
                <span role="cell">{row.env}</span>
                <span role="cell">{row.files}</span>
                <span role="cell">{row.checkpoints}</span>
              </div>
            ))}
          </div>
        ) : null}
        <div className="support-checklist compact-stack">
          <article className="support-check-item">
            <strong>部署摘要</strong>
            <p>
              {props.deploymentStatus
                ? `目录缺失 ${props.deploymentStatus.summary.missing_directories} / 环境缺失 ${props.deploymentStatus.summary.missing_conda_envs} / 必需文件缺失 ${props.deploymentStatus.summary.missing_required_files} / 警告 ${props.deploymentStatus.summary.warnings}`
                : props.deploymentError || "尚未读取远端部署状态。"}
            </p>
          </article>
          <article className="support-check-item">
            <strong>主线环境</strong>
            <p>{props.deploymentStatus ? formatDeploymentEnvSummary(props.deploymentStatus) : "读取后会显示 mast3r / monst3r / spann3r / align3r / fast3r / cut3r。"}</p>
          </article>
          <article className="support-check-item">
            <strong>目录与 README</strong>
            <p>{formatDeploymentDirectoryStatus(props.deploymentStatus)}</p>
          </article>
          <article className="support-check-item">
            <strong>缓存状态</strong>
            <p>{formatDeploymentCacheStatus(props.deploymentStatus)}</p>
          </article>
          {props.deploymentStatus?.cache?.last_error ? (
            <article className="support-check-item">
              <strong>最近错误</strong>
              <p>{props.deploymentStatus.cache.last_error}</p>
            </article>
          ) : null}
          <div className="service-card-actions">
            <button className="ghost-button small" onClick={() => void props.loadDeploymentStatus(true, true)} disabled={props.deploymentLoading} type="button">
              {props.deploymentLoading ? "检查中..." : "刷新远端部署状态"}
            </button>
          </div>
        </div>
      </article>

      <article className="panel">
        <PanelTitle eyebrow="Advisor" title="AI 配置状态" />
        <div className="support-checklist compact-stack">
          <article className="support-check-item">
            <strong>状态</strong>
            <p>{props.advisorReady ? `已配置：${props.advisorState.model}` : props.advisorState.message}</p>
          </article>
          <article className="support-check-item">
            <strong>建议位置</strong>
            <p>优先在任务完成或失败后用 AI 评估，再把结论放进汇报和下一轮实验计划。</p>
          </article>
          <button className="ghost-button" onClick={() => void props.openAdvisorSettings()} disabled={props.advisorConfigLoading} type="button">
            {props.advisorConfigLoading ? "读取中..." : "打开 AI 配置"}
          </button>
        </div>
      </article>

      <ModelRoadmapPanel activeModels={props.activeModelCatalog} deferredModels={props.deferredModelCatalog} />
      <SampleMatrixPanel
        samplesPayload={props.samplesPayload}
        errorMessage={props.samplesError}
        modelCatalog={props.modelCatalog}
        onLocateJob={(jobId) => props.openWorkspace("jobs", jobId)}
        onCopy={props.copyText}
      />

      <article className="panel">
        <PanelTitle eyebrow="Operator Notes" title="操作建议" />
        <div className="support-checklist compact-stack">
          {buildSystemChecklist().map((item) => (
            <article className="support-check-item" key={item.title}>
              <strong>{item.title}</strong>
              <p>{item.body}</p>
            </article>
          ))}
        </div>
      </article>
    </section>
  );
}
function ModelRoadmapPanel(props: {
  activeModels: ModelCatalogItem[];
  deferredModels: ModelCatalogItem[];
  compact?: boolean;
}) {
  const runnable = props.activeModels.filter((item) => item.runnable).length;
  const planned = props.activeModels.filter((item) => !item.runnable).length;

  return (
    <article className="panel model-roadmap-panel">
      <div className="section-head">
        <div>
          <span className="mini-label">模型路线</span>
          <h4>3R 接入与测评进度</h4>
          <p>当前主线集中在 MASt3R、MonST3R、Spann3R、Align3R、Fast3R、CUT3R；Pi3X、ZipMap、LingBot-Map 暂缓。</p>
        </div>
        <div className="model-roadmap-stats">
          <SummaryStat label="可运行" value={String(runnable)} />
          <SummaryStat label="待接入" value={String(planned)} />
        </div>
      </div>

      <div className="model-roadmap-list">
        {props.activeModels.map((item) => (
          <article className={`model-roadmap-item ${item.runnable ? "ready" : "planned"}`} key={item.value}>
            <div>
              <strong>{item.label}</strong>
              <p>{item.description}</p>
            </div>
            <div className="model-roadmap-meta">
              <span>{modelFamilyLabel(item.family)}</span>
              <span>{runnerStatusLabel(item.runner_status)}</span>
            </div>
          </article>
        ))}
      </div>

      {!props.compact && props.deferredModels.length > 0 ? (
        <div className="deferred-model-strip">
          <span className="mini-label">暂缓预研</span>
          <p>{props.deferredModels.map((item) => item.label).join(" / ")}</p>
        </div>
      ) : null}
    </article>
  );
}

function SampleMatrixPanel(props: {
  samplesPayload: SamplesPayload | null;
  errorMessage: string | null;
  modelCatalog: ModelCatalogItem[];
  onLocateJob?: (jobId: string) => void;
  onCopy?: (value: string, label: string) => Promise<void>;
  compact?: boolean;
}) {
  const [sortKey, setSortKey] = useState<SampleMatrixSortKey>("manifest");
  const [filterKey, setFilterKey] = useState<SampleMatrixFilterKey>("all");
  const [selectedSampleIds, setSelectedSampleIds] = useState<string[]>([]);
  const [batchLocateCursor, setBatchLocateCursor] = useState(0);
  const manifest = props.samplesPayload?.manifest ?? null;
  const summary = props.samplesPayload?.summary ?? null;
  const samples = manifest?.samples ?? [];
  const visibleSamples = props.compact ? samples.slice(0, 3) : samples;
  const scoringEntries = Object.entries(manifest?.scoring ?? {});
  const activeModels = manifest?.active_models ?? [];
  const deferredModels = manifest?.deferred_models ?? [];
  const compactScoringEntries = props.compact ? scoringEntries.slice(0, 3) : [];
  const statusCountEntries = Object.entries(summary?.status_counts ?? {}).sort(
    ([leftStatus, leftCount], [rightStatus, rightCount]) => rightCount - leftCount || leftStatus.localeCompare(rightStatus)
  );
  const jobMatrixRows = props.samplesPayload?.job_matrix?.rows ?? [];
  const unassignedJobs = props.samplesPayload?.job_matrix?.unassigned_jobs ?? [];
  const sampleRows = useMemo<SampleMatrixRowView[]>(() => {
    const matrix = new Map(jobMatrixRows.map((row) => [row.sample_id, row.jobs_by_model as Record<string, SampleMatrixJob>]));
    const rows = visibleSamples.map((sample, index) => {
      const jobsByModel = matrix.get(sample.id) ?? {};
      const requiredModels = sample.required_models ?? [];
      const requiredModelSet = new Set(requiredModels);
      const compareModels = Array.from(new Set([...(sample.required_models ?? []), ...(sample.optional_models ?? [])]));
      const rowStats = summarizeSampleMatrixRow(compareModels, jobsByModel);
      const rowScore = summarizeMatrixRowScore(compareModels, jobsByModel);
      return {
        sample,
        rowIndex: index,
        jobsByModel,
        requiredModels,
        requiredModelSet,
        compareModels,
        rowStats,
        rowScore
      };
    });
    const filteredRows = rows.filter((row) => matrixRowMatchesFilter(row, filterKey));
    return filteredRows.sort((left, right) => compareSampleMatrixRows(left, right, sortKey));
  }, [filterKey, jobMatrixRows, sortKey, visibleSamples]);
  const selectedSampleSet = useMemo(() => new Set(selectedSampleIds), [selectedSampleIds]);
  const selectedRows = useMemo(
    () => sampleRows.filter((row) => selectedSampleSet.has(row.sample.id)),
    [sampleRows, selectedSampleSet]
  );
  const batchTargetRows = useMemo(
    () => (selectedRows.length > 0 ? selectedRows : sampleRows),
    [sampleRows, selectedRows]
  );
  const batchTargetJobIds = useMemo(() => collectSampleMatrixJobIds(batchTargetRows), [batchTargetRows]);
  const allVisibleSelected = sampleRows.length > 0 && sampleRows.every((row) => selectedSampleSet.has(row.sample.id));
  const nextLocateIndex = batchTargetJobIds.length > 0 ? (batchLocateCursor % batchTargetJobIds.length) + 1 : 0;

  useEffect(() => {
    setSelectedSampleIds((current) => {
      const available = new Set(sampleRows.map((row) => row.sample.id));
      const next = current.filter((sampleId) => available.has(sampleId));
      return next.length === current.length ? current : next;
    });
  }, [sampleRows]);

  useEffect(() => {
    if (batchTargetJobIds.length === 0) {
      if (batchLocateCursor !== 0) {
        setBatchLocateCursor(0);
      }
      return;
    }
    if (batchLocateCursor >= batchTargetJobIds.length) {
      setBatchLocateCursor(0);
    }
  }, [batchLocateCursor, batchTargetJobIds.length]);

  function toggleSampleRowSelection(sampleId: string) {
    setSelectedSampleIds((current) =>
      current.includes(sampleId) ? current.filter((id) => id !== sampleId) : [...current, sampleId]
    );
    setBatchLocateCursor(0);
  }

  function toggleAllVisibleRows() {
    if (allVisibleSelected) {
      setSelectedSampleIds([]);
    } else {
      setSelectedSampleIds(sampleRows.map((row) => row.sample.id));
    }
    setBatchLocateCursor(0);
  }

  async function handleCopyBatchJobIds() {
    if (!props.onCopy || batchTargetJobIds.length === 0) {
      return;
    }
    const scopeLabel = selectedRows.length > 0 ? "选中样例任务ID" : "筛选样例任务ID";
    await props.onCopy(batchTargetJobIds.join("\n"), scopeLabel);
  }

  function handleLocateNextBatchJob() {
    if (!props.onLocateJob || batchTargetJobIds.length === 0) {
      return;
    }
    const targetIndex = batchLocateCursor % batchTargetJobIds.length;
    props.onLocateJob(batchTargetJobIds[targetIndex]);
    setBatchLocateCursor((targetIndex + 1) % batchTargetJobIds.length);
  }

  return (
    <article className="panel sample-matrix-panel">
      <div className="section-head">
        <div>
          <span className="mini-label">样例库</span>
          <h4>样例清单与测评矩阵</h4>
          <p>{manifest?.purpose ?? "等待 /api/samples 返回共享样例计划。"}</p>
        </div>
        <div className="sample-matrix-stats">
          <SummaryStat label="样例数" value={String(summary?.sample_count ?? samples.length)} />
          <SummaryStat label="活跃模型" value={String(activeModels.length)} />
          <SummaryStat label="评分维度" value={String(scoringEntries.length)} />
        </div>
      </div>

      {props.errorMessage && !props.samplesPayload ? <div className="empty-state">{props.errorMessage}</div> : null}

      {manifest ? (
        <>
          <div className="sample-matrix-meta">
            <div>
              <span className="mini-label">更新时间</span>
              <strong>{manifest.last_updated ?? "未标记"}</strong>
            </div>
            <div>
              <span className="mini-label">状态分布</span>
              <p>{formatCountMap(summary?.status_counts, sampleStatusLabel)}</p>
            </div>
            <div>
              <span className="mini-label">输入类型</span>
              <p>{formatCountMap(summary?.source_counts, sourceTypeLabel)}</p>
            </div>
          </div>

          <div className="sample-model-strip">
            <div>
              <span className="mini-label">主线模型</span>
              <p>{formatModelList(activeModels, props.modelCatalog)}</p>
            </div>
            {deferredModels.length > 0 ? (
              <div>
                <span className="mini-label">暂缓模型</span>
                <p>{formatModelList(deferredModels, props.modelCatalog)}</p>
              </div>
            ) : null}
          </div>

          {compactScoringEntries.length > 0 ? (
            <div className="scoring-compact-strip" aria-label="评分维度快览">
              {compactScoringEntries.map(([key, metrics]) => (
                <div key={key}>
                  <strong>{scoringCategoryLabel(key)}</strong>
                  <span>{metrics.length} 项指标</span>
                </div>
              ))}
            </div>
          ) : null}

          {visibleSamples.length > 0 ? (
            <section className="sample-compare-section" aria-label="样例对比视图">
              <div className="sample-compare-head">
                <div>
                  <span className="mini-label">样例对比</span>
                  <strong>直接对照样例编号、必跑模型和任务入口</strong>
                  <p>先看当前状态计数，再顺着 seed 任务跳到任务中心核对执行情况。</p>
                </div>
                <div className="sample-compare-side">
                  <div className="sample-status-strip" aria-label="当前状态计数">
                    {statusCountEntries.length > 0 ? (
                      statusCountEntries.map(([status, count]) => (
                        <div className="sample-status-pill" key={status}>
                          <span>{sampleStatusLabel(status)}</span>
                          <strong>{count}</strong>
                        </div>
                      ))
                    ) : (
                      <div className="sample-status-pill empty">
                        <span>当前状态计数</span>
                        <strong>暂无</strong>
                      </div>
                    )}
                  </div>
                  {!props.compact ? (
                    <div className="sample-compare-tools">
                      <label className="field compact">
                        <span>排序</span>
                        <select
                          value={sortKey}
                          onChange={(event) => {
                            setSortKey(event.target.value as SampleMatrixSortKey);
                            setBatchLocateCursor(0);
                          }}
                        >
                          <option value="manifest">按样例清单</option>
                          <option value="completion">按完成度</option>
                          <option value="score">按评分均值</option>
                          <option value="attention">按待处理优先</option>
                        </select>
                      </label>
                      <label className="field compact">
                        <span>筛选</span>
                        <select
                          value={filterKey}
                          onChange={(event) => {
                            setFilterKey(event.target.value as SampleMatrixFilterKey);
                            setBatchLocateCursor(0);
                          }}
                        >
                          <option value="all">全部样例</option>
                          <option value="attention">仅看待处理</option>
                          <option value="running">仅看运行中</option>
                          <option value="unfinished">仅看未完成</option>
                        </select>
                      </label>
                    </div>
                  ) : null}
                </div>
              </div>

              {!props.compact ? (
                <div className="sample-bulk-strip" aria-label="样例矩阵批量操作">
                  <div className="sample-bulk-summary">
                    <span className="mini-label">批量操作</span>
                    <strong>
                      {selectedRows.length > 0
                        ? `已选 ${selectedRows.length} 行样例`
                        : `未选择行，默认作用于当前筛选 ${sampleRows.length} 行`}
                    </strong>
                    <p>{batchTargetJobIds.length > 0 ? `可操作任务ID：${batchTargetJobIds.length}` : "当前范围还没有任务ID。"}</p>
                  </div>
                  <div className="sample-bulk-actions">
                    <button className="ghost-button small" onClick={toggleAllVisibleRows} type="button">
                      {allVisibleSelected ? "取消全选" : "全选筛选行"}
                    </button>
                    <button
                      className="ghost-button small"
                      disabled={!props.onCopy || batchTargetJobIds.length === 0}
                      onClick={() => void handleCopyBatchJobIds()}
                      type="button"
                    >
                      复制任务ID
                    </button>
                    <button
                      className="ghost-button small"
                      disabled={!props.onLocateJob || batchTargetJobIds.length === 0}
                      onClick={handleLocateNextBatchJob}
                      type="button"
                    >
                      定位下一条
                    </button>
                    {batchTargetJobIds.length > 0 ? (
                      <span className="sample-bulk-cursor">
                        游标 {nextLocateIndex}/{batchTargetJobIds.length}
                      </span>
                    ) : null}
                  </div>
                </div>
              ) : null}

              <div className="sample-compare-grid">
                {sampleRows.map((row) => {
                  const { sample, jobsByModel, requiredModels, requiredModelSet, compareModels, rowStats, rowScore } = row;
                  return (
                    <article className="sample-compare-row" key={sample.id}>
                      <div className="sample-compare-main">
                        <div className="sample-card-head">
                          <div className="sample-card-head-main">
                            {!props.compact ? (
                              <input
                                className="sample-row-checkbox"
                                type="checkbox"
                                checked={selectedSampleSet.has(sample.id)}
                                onChange={() => toggleSampleRowSelection(sample.id)}
                                aria-label={`选择样例 ${sample.id}`}
                              />
                            ) : null}
                            <strong>{sample.id}</strong>
                          </div>
                          <span className="status-badge">{sampleStatusLabel(sample.status)}</span>
                        </div>
                        <p>{sample.purpose}</p>
                        <div className="sample-card-meta">
                          <span>{sourceTypeLabel(sample.source_type)}</span>
                          <span>{sample.target_file_count ? `${sample.target_file_count} 个文件` : `${sample.target_duration_seconds ?? "-"} 秒`}</span>
                        </div>
                        <div className="sample-row-progress" aria-label={`${sample.id} 执行进度`}>
                          <div className="sample-row-progress-head">
                            <span>矩阵完成度</span>
                            <strong>
                              {rowStats.finished}/{rowStats.total}
                            </strong>
                          </div>
                          <div className="sample-row-progress-track">
                            <div className="sample-row-progress-fill" style={{ width: `${rowStats.completionPercent}%` }} />
                          </div>
                          <div className="sample-row-progress-meta">
                            <span>运行 {rowStats.running}</span>
                            <span>待处理 {rowStats.attention}</span>
                            <span>待派发 {rowStats.pending}</span>
                            <span>缺失 {rowStats.missing}</span>
                            <span>{rowScore.metricCount > 0 ? `均分 ${rowScore.average?.toFixed(2)}` : "均分 --"}</span>
                          </div>
                        </div>
                        <div className="sample-card-models sample-required-models">
                          {requiredModels.length > 0 ? (
                            requiredModels.map((model) => <span key={model}>{modelDisplayName(model, props.modelCatalog)}</span>)
                          ) : (
                            <span>未标记必跑模型</span>
                          )}
                        </div>
                        <p className="sample-compare-status-copy">
                          {sampleStatusLabel(sample.status)}
                          {sample.seed_job_id ? " · 已关联 seed 任务" : " · 尚未关联 seed 任务"}
                        </p>
                        <div className={`sample-seed-callout ${sample.seed_job_id ? "available" : "missing"}`}>
                          <span className="mini-label">任务定位</span>
                          {sample.seed_job_id ? (
                            <>
                              <strong>{sample.seed_job_id}</strong>
                              <p>任务中心可直接按这个 seed 任务继续核对日志、状态和结果。</p>
                              {props.onLocateJob ? (
                                <button className="ghost-button small sample-locate-button" onClick={() => props.onLocateJob?.(sample.seed_job_id!)} type="button">
                                  定位到任务中心
                                </button>
                              ) : null}
                            </>
                          ) : (
                            <p>当前还没有 seed 任务，先选样例或创建首条基准任务。</p>
                          )}
                        </div>
                      </div>

                      <div className="sample-compare-matrix">
                        <div className="sample-compare-matrix-head">
                          <span className="mini-label">模型执行矩阵</span>
                          <p>每格包含任务状态、评分快照和核心产物提示，方便横向对照。</p>
                        </div>
                        {compareModels.length > 0 ? (
                          <div className="sample-compare-matrix-grid">
                            {compareModels.map((model) => {
                              const job = jobsByModel[model] as SampleMatrixJob | undefined;
                              const cellState = job?.status ?? "missing";
                              const scoreDigest = summarizeScoreSnapshot(job?.score_snapshot);
                              return (
                                <article className={`sample-model-cell ${cellState}`} key={`${sample.id}-${model}`}>
                                  <div className="sample-model-cell-head">
                                    <strong>{modelDisplayName(model, props.modelCatalog)}</strong>
                                    <span className={`sample-model-state ${cellState}`}>{job ? job.status_label : "未跑"}</span>
                                  </div>
                                  <p>{job?.progress_message || (job ? `阶段：${job.phase}` : "尚未创建对应任务。")}</p>
                                  <div className="sample-model-cell-meta">
                                    <span>{requiredModelSet.has(model) ? "Required" : "Optional"}</span>
                                    <span>{primaryArtifactHint(job?.primary_artifacts)}</span>
                                  </div>
                                  <div className={`sample-score-signal ${scoreDigest.tone}`}>
                                    <div>
                                      <strong>{scoreDigest.label}</strong>
                                      <span>{scoreDigest.metricCount > 0 ? `${scoreDigest.metricCount} 项指标` : "暂无评分数据"}</span>
                                    </div>
                                    <strong>{scoreDigest.percent > 0 ? `${scoreDigest.percent}%` : "--"}</strong>
                                  </div>
                                  <div className="sample-score-track" aria-hidden>
                                    <div className={`sample-score-fill ${scoreDigest.tone}`} style={{ width: `${scoreDigest.percent}%` }} />
                                  </div>
                                  <div className="sample-model-cell-meta">
                                    <span>{job?.job_id ? `任务：${job.job_id}` : "尚无任务记录"}</span>
                                    {job?.job_id && props.onLocateJob ? (
                                      <button className="ghost-button small sample-locate-button" onClick={() => props.onLocateJob?.(job.job_id)} type="button">
                                        定位任务
                                      </button>
                                    ) : null}
                                  </div>
                                </article>
                              );
                            })}
                          </div>
                        ) : (
                          <div className="empty-state">暂无模型矩阵</div>
                        )}
                      </div>
                    </article>
                  );
                })}
              </div>
              {sampleRows.length === 0 ? <div className="empty-state">当前排序/筛选下暂无样例。</div> : null}
            </section>
          ) : (
            <div className="empty-state">样例清单还没有条目。</div>
          )}

          {!props.compact && scoringEntries.length > 0 ? (
            <div className="scoring-grid">
              {scoringEntries.map(([key, metrics]) => (
                <article className="scoring-card" key={key}>
                  <strong>{scoringCategoryLabel(key)}</strong>
                  <p>{metrics.map((metric) => metricLabel(metric)).join(" / ")}</p>
                </article>
              ))}
            </div>
          ) : null}

          {!props.compact && unassignedJobs.length > 0 ? (
            <section className="sample-unassigned-panel" aria-label="未归档任务池">
              <div className="sample-unassigned-head">
                <div>
                  <span className="mini-label">未归档任务池</span>
                  <strong>{unassignedJobs.length} 条任务还没有绑定样例</strong>
                </div>
                {props.onCopy ? (
                  <button
                    className="ghost-button small"
                    onClick={() => void props.onCopy?.(unassignedJobs.map((job) => job.job_id).join("\n"), "未归档任务ID")}
                    type="button"
                  >
                    复制ID
                  </button>
                ) : null}
              </div>
              <div className="sample-unassigned-grid">
                {unassignedJobs.slice(0, 8).map((job) => (
                  <article className={`sample-unassigned-card ${job.status}`} key={job.job_id}>
                    <div>
                      <strong>{job.job_id}</strong>
                      <span>{modelDisplayName(job.model, props.modelCatalog)} · {job.status_label}</span>
                    </div>
                    <p>{job.progress_message || `阶段：${job.phase}`}</p>
                    {props.onLocateJob ? (
                      <button className="ghost-button small" onClick={() => props.onLocateJob?.(job.job_id)} type="button">
                        定位任务
                      </button>
                    ) : null}
                  </article>
                ))}
              </div>
            </section>
          ) : null}
        </>
      ) : null}
    </article>
  );
}

function EvaluationPanel(props: {
  evaluation: EvaluationPayload | null;
  jobId: string;
  saving: boolean;
  onSave: (jobId: string, payload: EvaluationPayload) => Promise<void>;
}) {
  const [draft, setDraft] = useState<EvaluationPayload>({ job_id: props.jobId, notes: "" });

  useEffect(() => {
    setDraft({
      job_id: props.jobId,
      structure_completeness: props.evaluation?.structure_completeness ?? null,
      trajectory_stability: props.evaluation?.trajectory_stability ?? null,
      noise: props.evaluation?.noise ?? props.evaluation?.noise_control ?? null,
      dynamic_handling: props.evaluation?.dynamic_handling ?? null,
      depth_continuity: props.evaluation?.depth_continuity ?? props.evaluation?.depth_consistency ?? null,
      presentation_usability: props.evaluation?.presentation_usability ?? null,
      notes: props.evaluation?.notes ?? ""
    });
  }, [props.evaluation, props.jobId]);

  const fields: Array<{ key: keyof EvaluationPayload; label: string }> = [
    { key: "structure_completeness", label: "结构完整性" },
    { key: "trajectory_stability", label: "轨迹稳定性" },
    { key: "noise", label: "噪声控制" },
    { key: "dynamic_handling", label: "动态处理" },
    { key: "depth_continuity", label: "深度连续性" },
    { key: "presentation_usability", label: "展示可用性" }
  ];

  return (
    <form
      className="evaluation-form"
      onSubmit={(event) => {
        event.preventDefault();
        void props.onSave(props.jobId, draft);
      }}
    >
      <div className="evaluation-grid">
        {fields.map((field) => (
          <label className="field" key={String(field.key)}>
            <span>{field.label}</span>
            <select
              value={draft[field.key] == null ? "" : String(draft[field.key])}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  [field.key]: event.target.value ? Number(event.target.value) : null
                }))
              }
            >
              <option value="">未评分</option>
              <option value="1">1</option>
              <option value="2">2</option>
              <option value="3">3</option>
              <option value="4">4</option>
              <option value="5">5</option>
            </select>
          </label>
        ))}
      </div>
      <label className="field">
        <span>备注</span>
        <textarea
          rows={4}
          value={draft.notes ?? ""}
          onChange={(event) => setDraft((current) => ({ ...current, notes: event.target.value }))}
          placeholder="记录结构问题、轨迹漂移、动态区域表现、适不适合展示。"
        />
      </label>
      <div className="evaluation-actions">
        <span className="muted-text">
          {props.evaluation?.updated_at ? `上次保存：${formatDateTime(props.evaluation.updated_at)}` : "尚未保存人工评分。"}
        </span>
        <button className="ghost-button small" disabled={props.saving} type="submit">
          {props.saving ? "保存中..." : "保存评分"}
        </button>
      </div>
    </form>
  );
}

function JobDetail(props: {
  selectedJob: JobPayload;
  advisorState: AdvisorStatus;
  actionKey: string | null;
  savingEvaluation: boolean;
  canDispatch: boolean;
  running: boolean;
  assetUrl: (path: string) => string;
  onAction: (path: string, key: string) => Promise<void>;
  onSaveEvaluation: (jobId: string, payload: EvaluationPayload) => Promise<void>;
  onConfigureAdvisor: () => void;
  onOpenOutput: (relativePath: string) => Promise<void>;
  onPreviewAsset: (asset: PreviewAsset) => void;
  onCopy: (value: string, label: string) => Promise<void>;
}) {
  const job = props.selectedJob.job;
  const summary = props.selectedJob.result_summary;
  const latestLogLine = getLatestLogLine(props.selectedJob.logs);
  const criticalLogLine = getCriticalLogLine(props.selectedJob.logs);
  const [logQuery, setLogQuery] = useState("");
  const normalizedLogQuery = logQuery.trim().toLowerCase();
  const filteredLogs = useMemo(() => {
    if (!normalizedLogQuery) {
      return props.selectedJob.logs;
    }
    return props.selectedJob.logs.filter((log) => {
      const haystack = [log.name, log.relative_path, log.tail || ""].join("\n").toLowerCase();
      return haystack.includes(normalizedLogQuery);
    });
  }, [normalizedLogQuery, props.selectedJob.logs]);
  const outputSections = buildOutputSections(props.selectedJob.outputs, job.model);
  const progress = props.selectedJob.phase_display;
  const advisorSuggested = isAdvisorSuggested(job.status);
  const advisorReport = props.selectedJob.advisor_report ?? null;

  useEffect(() => {
    setLogQuery("");
  }, [job.job_id]);

  function scrollToInspectorSection(sectionId: string) {
    const target = document.getElementById(sectionId);
    if (!target) {
      return;
    }
    target.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  return (
    <div className="detail-stack">
      <div className={`detail-hero ${job.status}`}>
        <div className="detail-hero-head">
          <div className="hero-copy">
            <div className="hero-badges">
              <StatusBadge state={job.status} label={statusLabel(job.status)} />
              <span className="hero-tag">{statusModelLabel(job.model)}</span>
              <span className="hero-tag">{sourceTypeLabel(job.source_type)}</span>
            </div>
            <h3>{progress.label}</h3>
            <p>{job.progress_message || progress.description}</p>
          </div>
          <div className="detail-score-block">
            <div className="detail-score">{progress.percent}%</div>
            <span>{currentStepLabel(progress.steps)}</span>
          </div>
        </div>

        <div className="hero-progress">
          <div className="hero-progress-track">
            <div className="hero-progress-fill" style={{ width: `${progress.percent}%` }} />
          </div>
          <div className="hero-progress-labels">
            <span>当前阶段：{progress.label}</span>
            <span>{job.job_id}</span>
          </div>
        </div>

        <div className="step-grid">
          {progress.steps.map((step, index) => (
            <article className={`step-card ${step.state}`} key={step.code}>
              <span className="step-index">{index + 1}</span>
              <strong>{step.label}</strong>
              <p>{step.hint}</p>
            </article>
          ))}
        </div>
      </div>

      <article className={`advisor-recommendation ${advisorSuggested ? "active" : ""}`}>
        <div>
          <span className="mini-label">AI 评估建议</span>
          <strong>
            {!props.advisorState.enabled
              ? "AI 评估未启用"
              : !props.advisorState.configured
                ? "AI 评估配置还没补齐"
                : advisorReport
                  ? "当前任务已经有 AI 评估结果"
                  : advisorSuggested
                    ? "现在就适合做 AI 评估"
                    : "建议先等任务跑完或出现错误"}
          </strong>
          <p>
            {!props.advisorState.enabled || !props.advisorState.configured
              ? props.advisorState.message
              : advisorReport
                ? "可以直接查看问题、下一步建议和汇报话术；如果你又重跑了一版，再点一次会刷新。"
                : advisorSuggested
                  ? "当前状态已经足够让 AI 基于参数、scene_meta、摘要和日志给出判断。"
                  : "运行中的任务更适合先看阶段卡片和最新日志，结果稳定后再让 AI 做总结。"}
          </p>
        </div>
        <div className="advisor-workbench-actions">
          {!props.advisorState.enabled || !props.advisorState.configured ? (
            <button onClick={props.onConfigureAdvisor} type="button">
              配置 AI
            </button>
          ) : null}
          {advisorReport?.teacher_talk ? (
            <button onClick={() => void props.onCopy(advisorReport.teacher_talk, "汇报话术")} type="button">
              复制汇报话术
            </button>
          ) : null}
          <button
            disabled={!props.advisorState.enabled || !props.advisorState.configured || !advisorSuggested || props.actionKey === "advisor"}
            onClick={() => props.onAction(`/api/jobs/${job.job_id}/advisor/evaluate`, "advisor")}
            type="button"
          >
            {props.actionKey === "advisor" ? "评估中..." : "AI评估"}
          </button>
        </div>
      </article>

      <div className="action-row">
        <button
          disabled={!props.canDispatch || props.actionKey === "dispatch"}
          onClick={() => props.onAction(`/api/jobs/${job.job_id}/dispatch`, "dispatch")}
          type="button"
        >
          运行
        </button>
        <button
          disabled={props.running || props.actionKey === "retry"}
          onClick={() => props.onAction(`/api/jobs/${job.job_id}/retry`, "retry")}
          type="button"
        >
          重试
        </button>
        <button
          disabled={props.actionKey === "duplicate"}
          onClick={() => props.onAction(`/api/jobs/${job.job_id}/duplicate`, "duplicate")}
          type="button"
        >
          复制
        </button>
        <button
          className="danger"
          disabled={!props.running || props.actionKey === "cancel"}
          onClick={() => props.onAction(`/api/jobs/${job.job_id}/cancel`, "cancel")}
          type="button"
        >
          取消
        </button>
      </div>

      <div className="inspector-nav-strip">
        <span className="mini-label">检查器导航</span>
        <div className="inspector-nav-actions">
          <button className="ghost-button small" type="button" onClick={() => scrollToInspectorSection("job-summary-panel")}>
            摘要
          </button>
          <button className="ghost-button small" type="button" onClick={() => scrollToInspectorSection("job-outputs-panel")}>
            输出
          </button>
          <button className="ghost-button small" type="button" onClick={() => scrollToInspectorSection("job-logs-panel")}>
            日志
          </button>
          <button className="ghost-button small" type="button" onClick={() => scrollToInspectorSection("job-evaluation-panel")}>
            人工评分
          </button>
          <button className="ghost-button small" type="button" onClick={() => scrollToInspectorSection("job-ai-panel")}>
            AI评估
          </button>
          <button className="ghost-button small" type="button" onClick={() => scrollToInspectorSection("job-inputs-panel")}>
            输入
          </button>
        </div>
      </div>

      {job.error_message ? (
        <article className="soft-panel error-panel">
          <h4>错误原因</h4>
          <p>{job.error_message}</p>
        </article>
      ) : null}

      <div className="detail-inspector-grid">
        <section className="detail-inspector-main">
          <article className="soft-panel inspector-panel" id="job-summary-panel">
            <div className="section-head">
              <div>
                <h4>结果摘要</h4>
                <p>先看核心统计，再看检查对象与下一步建议。</p>
              </div>
            </div>
            <SummaryPanel summary={props.selectedJob.result_summary} />
          </article>

          <article className="soft-panel inspector-panel" id="job-outputs-panel">
            <div className="section-head">
              <div>
                <h4>输出结果</h4>
                <p>按用途分组展示，不再把 MonST3R / DUSt3R 产物全都摊成一排。</p>
              </div>
              <span className="section-pill">{props.selectedJob.outputs.length} 个文件</span>
            </div>
            {outputSections.length > 1 ? (
              <div className="output-anchor-strip">
                {outputSections.map((section) => (
                  <button
                    className="ghost-button small"
                    key={section.key}
                    onClick={() => scrollToInspectorSection(`job-output-group-${section.key}`)}
                    type="button"
                  >
                    {section.title}
                  </button>
                ))}
                <button className="ghost-button small" onClick={() => scrollToInspectorSection("job-logs-panel")} type="button">
                  跳到日志
                </button>
              </div>
            ) : null}
            {outputSections.length > 0 ? (
              <div className="output-section-list">
                {outputSections.map((section) => (
                  <details className={`output-section ${section.accent}`} id={`job-output-group-${section.key}`} key={section.key} open={section.defaultOpen}>
                    <summary>
                      <div>
                        <strong>{section.title}</strong>
                        <p>{section.description}</p>
                      </div>
                      <span className="section-pill">{section.items.length}</span>
                    </summary>
                    <div className="output-grid">
                      {section.items.map((output) => (
                        <article className="output-card" key={output.relative_path}>
                          {output.is_image ? (
                            <button
                              className="output-preview-button"
                              type="button"
                              onClick={() =>
                                props.onPreviewAsset({
                                  url: props.assetUrl(output.url),
                                  name: output.display_name,
                                  kind: "image",
                                  note:
                                    section.key === "masks"
                                      ? "这是黑白掩膜/中间结果，不是最终彩色重建。MonST3R 的主要成果请优先看 scene.glb、相机轨迹和 frame_*.png。"
                                      : "这是图像产物预览。"
                                })
                              }
                            >
                              <img className="output-preview" src={props.assetUrl(output.url)} alt={output.display_name} />
                            </button>
                          ) : output.is_video ? (
                            <button
                              className="output-preview-button"
                              type="button"
                              onClick={() =>
                                props.onPreviewAsset({
                                  url: props.assetUrl(output.url),
                                  name: output.display_name,
                                  kind: "video",
                                  note: "这是视频产物预览。"
                                })
                              }
                            >
                              <div className="output-preview placeholder">VIDEO</div>
                            </button>
                          ) : (
                            <div className="output-preview placeholder">
                              {output.is_pointcloud ? "PLY" : output.is_model3d ? "GLB" : fileExtensionLabel(output.display_name)}
                            </div>
                          )}
                          <div>
                            <strong>{output.display_name}</strong>
                            <p>{describeOutput(output.display_name)}</p>
                            <div className="output-actions">
                              {output.is_image || output.is_video ? (
                                <button
                                  onClick={() =>
                                    props.onPreviewAsset({
                                      url: props.assetUrl(output.url),
                                      name: output.display_name,
                                      kind: output.is_video ? "video" : "image",
                                      note:
                                        section.key === "masks"
                                          ? "这是黑白掩膜/中间结果，不是最终彩色重建。"
                                          : output.is_video
                                            ? "这是视频产物预览。"
                                            : "这是图像产物预览。"
                                    })
                                  }
                                  type="button"
                                >
                                  预览
                                </button>
                              ) : null}
                              <a href={props.assetUrl(output.url)} download>
                                下载
                              </a>
                              <button onClick={() => props.onOpenOutput(output.relative_path)} type="button">
                                本地打开
                              </button>
                            </div>
                          </div>
                        </article>
                      ))}
                    </div>
                  </details>
                ))}
              </div>
            ) : (
              <span className="muted-text">结果回传后会出现在这里。</span>
            )}
          </article>

          <article className="soft-panel inspector-panel" id="job-logs-panel">
            <div className="section-head">
              <div>
                <h4>日志</h4>
                <p>优先看最新一条有效进展，再往下翻完整日志尾部。</p>
              </div>
              <div className="logs-head-actions">
                <span className="section-pill">
                  {normalizedLogQuery ? `命中 ${filteredLogs.length}/${props.selectedJob.logs.length}` : `${props.selectedJob.logs.length} 份`}
                </span>
                {latestLogLine ? (
                  <button className="ghost-button small" onClick={() => void props.onCopy(latestLogLine, "最新日志")} type="button">
                    复制最新
                  </button>
                ) : null}
                {criticalLogLine ? (
                  <button className="ghost-button small" onClick={() => void props.onCopy(criticalLogLine, "可疑日志")} type="button">
                    复制可疑行
                  </button>
                ) : null}
              </div>
            </div>
            {latestLogLine ? <div className="latest-log-banner">{latestLogLine}</div> : null}
            {criticalLogLine ? <div className="critical-log-banner">可疑日志：{criticalLogLine}</div> : null}
            <div className="log-filter-strip">
              <label className="field compact log-filter-field">
                <span>关键词</span>
                <input
                  type="search"
                  value={logQuery}
                  onChange={(event) => setLogQuery(event.target.value)}
                  placeholder="例如 error / timeout / cuda / oom"
                />
              </label>
              {normalizedLogQuery ? (
                <button className="ghost-button small" onClick={() => setLogQuery("")} type="button">
                  清除
                </button>
              ) : null}
            </div>
            {filteredLogs.length > 0 ? (
              <div className="log-list">
                {filteredLogs.map((log) => (
                  <div className="log-card" key={log.relative_path}>
                    <strong>{log.name}</strong>
                    <pre>{log.tail || "暂无日志。"}</pre>
                  </div>
                ))}
              </div>
            ) : props.selectedJob.logs.length > 0 ? (
              <div className="empty-state">没有匹配“{logQuery.trim()}”的日志。</div>
            ) : (
              <span className="muted-text">还没有日志。</span>
            )}
          </article>
        </section>

        <aside className="detail-inspector-rail">
          <article className="soft-panel inspector-panel">
            <div className="section-head">
              <div>
                <h4>检查器快照</h4>
                <p>快速核对时间、产物数量和最新进展。</p>
              </div>
            </div>
            <div className="meta-grid inspector-meta-grid">
              <MetaCard label="创建时间" value={formatDateTime(job.created_at)} />
              <MetaCard label="输入数量" value={String(summary?.inputs?.count ?? job.input_items.length ?? 0)} />
              <MetaCard label="回传产物" value={String(summary?.artifacts?.length ?? props.selectedJob.outputs.length)} />
              <MetaCard label="最新日志" value={latestLogLine || "尚无有效日志"} compact />
            </div>
          </article>

          <article className="soft-panel inspector-panel" id="job-evaluation-panel">
            <h4>人工评分</h4>
            <EvaluationPanel
              evaluation={props.selectedJob.evaluation ?? null}
              jobId={job.job_id}
              saving={props.savingEvaluation}
              onSave={props.onSaveEvaluation}
            />
          </article>

          <article className="soft-panel inspector-panel" id="job-ai-panel">
            <h4>AI 评估</h4>
            <AdvisorPanel report={advisorReport} />
          </article>

          <article className="soft-panel inspector-panel" id="job-inputs-panel">
            <h4>输入</h4>
            <div className="preview-grid">
              {props.selectedJob.previews.length > 0 ? (
                props.selectedJob.previews.map((preview) => (
                  <button
                    key={preview.relative_path}
                    className="preview-card"
                    type="button"
                    onClick={() =>
                      props.onPreviewAsset({
                        url: props.assetUrl(preview.url),
                        name: preview.display_name,
                        kind: "image",
                        note: "这是输入预览，不会跳转离开主界面。"
                      })
                    }
                  >
                    {preview.is_image ? <img src={props.assetUrl(preview.url)} alt={preview.display_name} /> : null}
                    <span>{preview.display_name}</span>
                  </button>
                ))
              ) : (
                <span className="muted-text">暂无输入预览。</span>
              )}
            </div>
          </article>
        </aside>
      </div>
    </div>
  );
}

function MetaCard(props: { label: string; value: string; compact?: boolean }) {
  return (
    <article className={`meta-card ${props.compact ? "compact" : ""}`}>
      <span>{props.label}</span>
      <strong>{props.value}</strong>
    </article>
  );
}

function currentStepLabel(steps: JobPayload["phase_display"]["steps"]) {
  const currentIndex = steps.findIndex((step) => step.state === "current");
  if (currentIndex >= 0) {
    return `第 ${currentIndex + 1} / ${steps.length} 阶段`;
  }
  if (steps.every((step) => step.state === "done")) {
    return "全部阶段完成";
  }
  return `共 ${steps.length} 个阶段`;
}

function getLatestLogLine(logs: JobPayload["logs"]) {
  for (const log of logs) {
    const lines = (log.tail || "")
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
    for (let index = lines.length - 1; index >= 0; index -= 1) {
      const line = lines[index];
      if (!/futurewarning|warning, cannot find cuda-compiled version of rope2d/i.test(line)) {
        return line;
      }
    }
  }
  return "";
}

function getCriticalLogLine(logs: JobPayload["logs"]) {
  const criticalPattern = /traceback|exception|fatal|runtimeerror|oom|cuda out of memory|failed|error/i;
  const ignoredPattern = /futurewarning|warning, cannot find cuda-compiled version of rope2d/i;
  for (const log of logs) {
    const lines = (log.tail || "")
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
    for (let index = lines.length - 1; index >= 0; index -= 1) {
      const line = lines[index];
      if (ignoredPattern.test(line)) {
        continue;
      }
      if (criticalPattern.test(line)) {
        return line;
      }
    }
  }
  return "";
}

function buildOutputSections(outputs: OutputItem[], model: string): OutputSection[] {
  const buckets: Record<string, OutputSection> = {
    main: {
      key: "main",
      title: "核心成果",
      description: model === "monst3r" ? "优先查看三维场景、点云和主要导出结果。" : "优先查看重建点云和主要可视化结果。",
      accent: "blue",
      defaultOpen: true,
      items: []
    },
    camera: {
      key: "camera",
      title: "相机与轨迹",
      description: "相机内参、位姿、轨迹等文本产物。",
      accent: "slate",
      defaultOpen: true,
      items: []
    },
    masks: {
      key: "masks",
      title: "掩膜与动态区域",
      description: "黑白动态 mask、扩张 mask 等辅助可视化，不是最终彩色重建。",
      accent: "gold",
      defaultOpen: false,
      items: []
    },
    confidence: {
      key: "confidence",
      title: "置信度与数组",
      description: "置信图、深度数组和中间数值文件。",
      accent: "slate",
      defaultOpen: false,
      items: []
    },
    visuals: {
      key: "visuals",
      title: "图像可视化",
      description: "可直接浏览的 PNG/JPG 结果图。",
      accent: "green",
      defaultOpen: true,
      items: []
    },
    other: {
      key: "other",
      title: "其他产物",
      description: "未归类但仍可下载查看的文件。",
      accent: "slate",
      defaultOpen: false,
      items: []
    }
  };

  outputs.forEach((output) => {
    const name = output.display_name.toLowerCase();
    if (output.is_model3d || output.is_pointcloud || /scene\.glb|pointcloud|matches\./i.test(name)) {
      buckets.main.items.push(output);
      return;
    }
    if (/traj|intrinsics|poses?|focal|camera/i.test(name) || name.endsWith(".txt")) {
      buckets.camera.items.push(output);
      return;
    }
    if (/mask/i.test(name)) {
      buckets.masks.items.push(output);
      return;
    }
    if (/conf|depth|frame_.*\.npy|\.npy$/i.test(name)) {
      buckets.confidence.items.push(output);
      return;
    }
    if (output.is_image) {
      buckets.visuals.items.push(output);
      return;
    }
    buckets.other.items.push(output);
  });

  const orderedKeys =
    model === "monst3r"
      ? ["main", "visuals", "camera", "masks", "confidence", "other"]
      : ["main", "visuals", "camera", "masks", "confidence", "other"];

  return orderedKeys
    .map((key) => buckets[key])
    .filter((section) => section && section.items.length > 0);
}

function Monst3rParams(props: {
  formState: FormState;
  updateFormField: (key: keyof FormState, value: string) => void;
}) {
  return (
    <div className="param-grid">
      {Object.keys(defaultMonst3rParams)
        .map((key) => (
          <ParamField
            key={key}
            name={key}
            value={props.formState[key as keyof FormState]}
            choices={monst3rParamChoices[key as keyof typeof defaultMonst3rParams]}
            onChange={(value) => props.updateFormField(key as keyof FormState, value)}
          />
        ))}
    </div>
  );
}

function ParamField(props: {
  name: string;
  value: string;
  choices?: ParamChoice[];
  onChange: (value: string) => void;
}) {
  const activeChoice = props.choices?.find((item) => item.value === props.value);

  if (props.choices && props.choices.length > 0) {
    return (
      <label className="field compact">
        <span>{formatParamLabel(props.name)}</span>
        <select value={props.value} onChange={(event) => props.onChange(event.target.value)}>
          {props.choices.map((choice) => (
            <option key={choice.value} value={choice.value}>
              {choice.label}
            </option>
          ))}
        </select>
        {activeChoice ? <small className="field-note">{activeChoice.note}</small> : null}
      </label>
    );
  }

  return (
    <label className="field compact">
      <span>{formatParamLabel(props.name)}</span>
      <input value={props.value} onChange={(event) => props.onChange(event.target.value)} />
    </label>
  );
}

function PanelTitle(props: { eyebrow: string; title: string }) {
  return (
    <div className="panel-title">
      <span>{props.eyebrow}</span>
      <h2>{props.title}</h2>
    </div>
  );
}

function MiniStat(props: { label: string; value: number | string }) {
  return (
    <div className="mini-stat">
      <span>{props.label}</span>
      <strong>{props.value}</strong>
    </div>
  );
}

function FilterPill(props: {
  active: boolean;
  count: number;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      className={`filter-pill ${props.active ? "active" : ""}`}
      onClick={props.onClick}
      type="button"
    >
      {props.label}
      <span>{props.count}</span>
    </button>
  );
}

function MessageBanner(props: { kind: "info" | "error"; message: string }) {
  return (
    <section className={`message-banner ${props.kind}`}>
      <strong>{props.kind === "info" ? "提示" : "需要处理"}</strong>
      <span>{props.message}</span>
    </section>
  );
}

function StatusBadge(props: { state: string; label: string }) {
  return <span className={`status-badge ${props.state}`}>{props.label}</span>;
}

function SummaryPanel(props: { summary: ResultSummary | null }) {
  if (!props.summary) {
    return <span className="muted-text">完成后自动生成摘要。</span>;
  }

  const highlights = props.summary.highlights ?? [];
  const nextActions = props.summary.next_actions ?? [];
  const artifactGroups = props.summary.artifact_groups ?? [];
  const primaryArtifacts = props.summary.primary_artifacts ?? [];
  const sceneMeta = props.summary.scene_meta ?? {};
  const sceneStats = [
    typeof sceneMeta["artifact_count"] === "number" ? { label: "远端整理", value: String(sceneMeta["artifact_count"]) } : null,
    typeof sceneMeta["glb_count"] === "number" ? { label: "GLB", value: String(sceneMeta["glb_count"]) } : null,
    typeof sceneMeta["frame_preview_count"] === "number" ? { label: "帧预览", value: String(sceneMeta["frame_preview_count"]) } : null,
    typeof sceneMeta["dynamic_mask_count"] === "number" ? { label: "动态Mask", value: String(sceneMeta["dynamic_mask_count"]) } : null,
    typeof sceneMeta["n_points"] === "number" ? { label: "点数", value: String(sceneMeta["n_points"]) } : null,
    typeof sceneMeta["input_count"] === "number" ? { label: "输入", value: String(sceneMeta["input_count"]) } : null
  ].filter(Boolean) as Array<{ label: string; value: string }>;

  return (
    <div className="summary-panel">
      <div className="summary-strip">
        <SummaryStat label="状态" value={props.summary.status_label} />
        <SummaryStat label="耗时" value={formatDuration(props.summary.duration_seconds ?? null)} />
        <SummaryStat label="输入" value={String(props.summary.inputs?.count ?? 0)} />
        <SummaryStat label="产物" value={String(props.summary.artifacts?.length ?? 0)} />
      </div>
      {sceneStats.length > 0 ? (
        <div className="summary-strip secondary">
          {sceneStats.map((item) => (
            <SummaryStat key={item.label} label={item.label} value={item.value} />
          ))}
        </div>
      ) : null}
      {primaryArtifacts.length > 0 ? (
        <div className="summary-primary-list">
          <strong>核心检查对象</strong>
          {primaryArtifacts.map((item) => (
            <div className="summary-primary-item" key={`${item.role}-${item.relative_path}`}>
              <span>{item.label || roleLabel(item.role)}</span>
              <p>{item.name}</p>
            </div>
          ))}
        </div>
      ) : null}
      {artifactGroups.length > 0 ? (
        <div className="summary-group-grid">
          {artifactGroups.map((item) => (
            <div className="summary-group-item" key={item.key}>
              <span>{item.label}</span>
              <strong>{item.count}</strong>
            </div>
          ))}
        </div>
      ) : null}
      {highlights.length > 0 ? (
        <ul>
          {highlights.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      ) : null}
      {nextActions.length > 0 ? (
        <ul>
          {nextActions.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

function roleLabel(role: string) {
  const labels: Record<string, string> = {
    scene: "三维场景",
    trajectory: "相机轨迹",
    intrinsics: "相机内参",
    frame_preview: "彩色帧预览",
    dynamic_mask: "动态区域",
    confidence: "置信数组"
  };
  return labels[role] ?? role.replace(/_/g, " ");
}

function summarizeScoreSnapshot(scoreSnapshot?: Record<string, number>) {
  if (!scoreSnapshot) {
    return {
      label: "评分快照：暂无",
      metricCount: 0,
      average: null as number | null,
      percent: 0,
      tone: "none" as "none" | "low" | "medium" | "high"
    };
  }
  const values = Object.values(scoreSnapshot).filter((value) => Number.isFinite(value));
  if (values.length === 0) {
    return {
      label: "评分快照：暂无",
      metricCount: 0,
      average: null as number | null,
      percent: 0,
      tone: "none" as "none" | "low" | "medium" | "high"
    };
  }
  const avg = values.reduce((sum, value) => sum + value, 0) / values.length;
  const bounded = Math.min(5, Math.max(0, avg));
  const tone: "low" | "medium" | "high" = bounded >= 4.2 ? "high" : bounded >= 3.2 ? "medium" : "low";
  return {
    label: `评分快照：${bounded.toFixed(2)}`,
    metricCount: values.length,
    average: bounded,
    percent: Math.round((bounded / 5) * 100),
    tone
  };
}

function summarizeSampleMatrixRow(compareModels: string[], jobsByModel: Record<string, SampleMatrixJob>) {
  const stats = {
    total: compareModels.length,
    finished: 0,
    running: 0,
    attention: 0,
    pending: 0,
    missing: 0,
    completionPercent: 0
  };

  compareModels.forEach((model) => {
    const job = jobsByModel[model];
    const status = job?.status ?? "missing";
    if (status === "finished") {
      stats.finished += 1;
      return;
    }
    if (status === "running" || status === "starting") {
      stats.running += 1;
      return;
    }
    if (status === "failed" || status === "cancelled") {
      stats.attention += 1;
      return;
    }
    if (status === "missing") {
      stats.missing += 1;
      return;
    }
    stats.pending += 1;
  });

  stats.completionPercent = stats.total > 0 ? Math.round((stats.finished / stats.total) * 100) : 0;
  return stats;
}

function summarizeMatrixRowScore(compareModels: string[], jobsByModel: Record<string, SampleMatrixJob>) {
  const digests = compareModels
    .map((model) => summarizeScoreSnapshot(jobsByModel[model]?.score_snapshot))
    .filter((digest) => typeof digest.average === "number");
  if (digests.length === 0) {
    return {
      average: null as number | null,
      percent: 0,
      metricCount: 0,
      tone: "none" as "none" | "low" | "medium" | "high"
    };
  }
  const totalAverage = digests.reduce((sum, digest) => sum + (digest.average ?? 0), 0);
  const average = totalAverage / digests.length;
  const metricCount = digests.reduce((sum, digest) => sum + digest.metricCount, 0);
  const bounded = Math.min(5, Math.max(0, average));
  return {
    average: bounded,
    percent: Math.round((bounded / 5) * 100),
    metricCount,
    tone: bounded >= 4.2 ? ("high" as const) : bounded >= 3.2 ? ("medium" as const) : ("low" as const)
  };
}

function matrixRowMatchesFilter(row: SampleMatrixRowView, filterKey: SampleMatrixFilterKey) {
  switch (filterKey) {
    case "attention":
      return row.rowStats.attention > 0;
    case "running":
      return row.rowStats.running > 0;
    case "unfinished":
      return row.rowStats.finished < row.rowStats.total;
    default:
      return true;
  }
}

function compareSampleMatrixRows(left: SampleMatrixRowView, right: SampleMatrixRowView, sortKey: SampleMatrixSortKey) {
  switch (sortKey) {
    case "completion": {
      const byCompletion = right.rowStats.completionPercent - left.rowStats.completionPercent;
      if (byCompletion !== 0) {
        return byCompletion;
      }
      const leftScore = left.rowScore.average ?? -1;
      const rightScore = right.rowScore.average ?? -1;
      const byScore = rightScore - leftScore;
      if (byScore !== 0) {
        return byScore;
      }
      break;
    }
    case "score": {
      const leftScore = left.rowScore.average ?? -1;
      const rightScore = right.rowScore.average ?? -1;
      const byScore = rightScore - leftScore;
      if (byScore !== 0) {
        return byScore;
      }
      const byCompletion = right.rowStats.completionPercent - left.rowStats.completionPercent;
      if (byCompletion !== 0) {
        return byCompletion;
      }
      break;
    }
    case "attention": {
      const byAttention = right.rowStats.attention - left.rowStats.attention;
      if (byAttention !== 0) {
        return byAttention;
      }
      const byRunning = right.rowStats.running - left.rowStats.running;
      if (byRunning !== 0) {
        return byRunning;
      }
      const leftPending = left.rowStats.pending + left.rowStats.missing;
      const rightPending = right.rowStats.pending + right.rowStats.missing;
      const byPending = rightPending - leftPending;
      if (byPending !== 0) {
        return byPending;
      }
      break;
    }
    case "manifest":
    default: {
      const byIndex = left.rowIndex - right.rowIndex;
      if (byIndex !== 0) {
        return byIndex;
      }
      break;
    }
  }
  return left.sample.id.localeCompare(right.sample.id);
}

function primaryArtifactHint(primaryArtifacts?: SampleMatrixJob["primary_artifacts"]) {
  if (!primaryArtifacts || primaryArtifacts.length === 0) {
    return "核心产物：暂无";
  }
  const first = primaryArtifacts[0];
  const label = first.label || roleLabel(first.role);
  return `核心产物：${label}`;
}

function collectSampleMatrixJobIds(rows: SampleMatrixRowView[]) {
  const ids: string[] = [];
  const seen = new Set<string>();
  for (const row of rows) {
    if (row.sample.seed_job_id && !seen.has(row.sample.seed_job_id)) {
      seen.add(row.sample.seed_job_id);
      ids.push(row.sample.seed_job_id);
    }
    for (const model of row.compareModels) {
      const jobId = row.jobsByModel[model]?.job_id;
      if (jobId && !seen.has(jobId)) {
        seen.add(jobId);
        ids.push(jobId);
      }
    }
  }
  return ids;
}

function isImageLikeFile(file: File) {
  return file.type.startsWith("image/") || /\.(png|jpe?g|bmp|webp)$/i.test(file.name);
}

function isVideoLikeFile(file: File) {
  return file.type.startsWith("video/") || /\.(mp4|mov|mkv|avi|webm)$/i.test(file.name);
}

function pendingFileRoleLabel(file: File) {
  if (isImageLikeFile(file)) {
    return "图片";
  }
  if (isVideoLikeFile(file)) {
    return "视频";
  }
  return "其他";
}

function formatFileSize(bytes: number) {
  if (bytes <= 0) {
    return "0 B";
  }
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  if (bytes < 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function SummaryStat(props: { label: string; value: string }) {
  return (
    <div className="summary-stat">
      <span>{props.label}</span>
      <strong>{props.value}</strong>
    </div>
  );
}

function AdvisorPanel(props: { report: AdvisorReport | null }) {
  if (!props.report) {
    return <span className="muted-text">点上方“AI评估”后，会基于参数、摘要、scene_meta 和日志生成建议。</span>;
  }

  return (
    <div className="advisor-panel">
      <div className="summary-strip">
        <SummaryStat label="评分" value={String(props.report.overall_score || "-")} />
        <SummaryStat label="结论" value={props.report.readiness || "-"} />
        <SummaryStat label="模型" value={props.report.advisor_model || "-"} />
      </div>
      <p className="advisor-summary">{props.report.summary}</p>
      {props.report.issues.length > 0 ? (
        <div>
          <strong>主要问题</strong>
          <ul>
            {props.report.issues.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {props.report.next_actions.length > 0 ? (
        <div>
          <strong>下一步</strong>
          <ul>
            {props.report.next_actions.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {props.report.teacher_talk ? (
        <div className="advisor-quote">
          <strong>可直接汇报</strong>
          <p>{props.report.teacher_talk}</p>
        </div>
      ) : null}
    </div>
  );
}

function AdvisorWorkbench(props: {
  job: JobPayload | null;
  advisorState: AdvisorStatus;
  actionKey: string | null;
  onEvaluate: (jobId: string) => void;
  onConfigure: () => void;
  onCopy: (value: string, label: string) => Promise<void>;
  compact?: boolean;
}) {
  if (!props.advisorState.enabled || !props.advisorState.configured) {
    return (
      <div className={`advisor-workbench ${props.compact ? "compact" : ""}`}>
        <div className="advisor-config-note">
          <strong>AI 评估暂不可用</strong>
          <p>{props.advisorState.message}</p>
          <div className="advisor-workbench-actions">
            <button onClick={props.onConfigure} type="button">
              立即配置
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!props.job) {
    return (
      <div className={`advisor-workbench ${props.compact ? "compact" : ""}`}>
        <div className="empty-state">
          先在“运行与结果”里选中一条任务，再来这里让 AI 做判断。通常优先评估已完成、失败或已取消的任务。
        </div>
      </div>
    );
  }

  const job = props.job;
  const report = job.advisor_report ?? null;
  const suggested = isAdvisorSuggested(job.job.status);

  return (
    <div className={`advisor-workbench ${props.compact ? "compact" : ""}`}>
      <div className="advisor-workbench-head">
        <div>
          <strong>{job.job.job_id}</strong>
          <p>
            {statusModelLabel(job.job.model)} · {statusLabel(job.job.status)} · {sourceTypeLabel(job.job.source_type)}
          </p>
        </div>
        <div className="advisor-workbench-actions">
          {report?.teacher_talk ? (
            <button onClick={() => void props.onCopy(report.teacher_talk, "汇报话术")} type="button">
              复制汇报话术
            </button>
          ) : null}
          <button
            disabled={!suggested || props.actionKey === "advisor"}
            onClick={() => props.onEvaluate(job.job.job_id)}
            type="button"
          >
            {props.actionKey === "advisor" ? "评估中..." : report ? "重新评估" : "开始评估"}
          </button>
        </div>
      </div>

      {!report ? (
        <div className="advisor-config-note">
          <strong>{suggested ? "现在适合调用 AI" : "暂时先别急着评估"}</strong>
          <p>
            {suggested
              ? "当前任务已经有足够信息可供总结，点右侧按钮就能得到问题诊断、下一步建议和汇报话术。"
              : "任务还在运行时，优先先看日志和阶段进度；等任务完成、失败或取消后再让 AI 做判断更有效。"}
          </p>
        </div>
      ) : (
        <>
          <AdvisorPanel report={report} />
          {props.compact ? null : (
            <div className="advisor-copy-row">
              <button onClick={() => void props.onCopy(report.summary, "AI 摘要")} type="button">
                复制摘要
              </button>
              {report.teacher_talk ? (
                <button onClick={() => void props.onCopy(report.teacher_talk, "汇报话术")} type="button">
                  复制汇报话术
                </button>
              ) : null}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function delay(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function friendlyError(error: unknown, fallback: string) {
  if (error instanceof Error && error.message) {
    if (/Failed to fetch|NetworkError|fetch/i.test(error.message)) {
      return "本地服务暂时不可用，请等顶部状态变成“就绪”。";
    }
    return error.message;
  }
  return fallback;
}

function serviceStatusLabel(state: ServiceState) {
  if (state === "ready") {
    return "就绪";
  }
  if (state === "degraded") {
    return "未连接";
  }
  return "启动中";
}

function backendStatusText(status: BackendStatusPayload | null) {
  if (!status) {
    return "正在检查桌面后端托管状态。";
  }
  if (status.running && status.managed_by_tauri) {
    return status.log_path ? `桌面端已自动启动后端。日志：${status.log_path}` : "桌面端已自动启动后端。";
  }
  if (status.running) {
    return "检测到已有后端，当前会直接复用。";
  }
  return status.message || "后端暂未启动。";
}

function isParamFieldKey(key: keyof FormState) {
  return key in defaultDust3rParams || key in defaultMonst3rParams;
}

function buildDust3rPreset(preset: PresetKey, fileCount: number): typeof defaultDust3rParams {
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

function buildMonst3rPreset(preset: PresetKey): typeof defaultMonst3rParams {
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

function presetLabel(preset: PresetKey | null) {
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

function allowedSourceTypesForModel(model: string) {
  if (model === "monst3r") {
    return ["video", "frames"];
  }
  return ["images"];
}

function inputHint(model: string, sourceType: string) {
  if (model === "monst3r" && sourceType === "video") {
    return "上传 1 个视频文件";
  }
  if (model === "monst3r") {
    return "上传连续帧图片，建议 3 张以上";
  }
  return model === "mast3r" ? "上传 2 张或更多同场景图片" : "上传 2 张或更多图片";
}

function buildCreateGuidance(model: string, sourceType: string, fileCount: number) {
  if (model === "monst3r" && sourceType === "video") {
    return [
      "先用 1 段短视频做样例测试，长度尽量控制在几十帧以内。",
      "推荐先用正式基线：图像尺寸 512、最大帧数 48、批大小 1；如果只想先验链路，再切到 224 / 24 帧。",
      fileCount === 1 ? "当前文件数量正确，可以直接创建任务。" : "视频模式请只放 1 个视频文件。"
    ];
  }
  if (model === "monst3r") {
    return [
      "帧序列建议选连续视角变化的小样本，先用 3 到 12 张测试。",
      "先用推荐基线跑第一版，只有在结果不错时再把帧数和窗口参数往上加。",
      fileCount >= 2 ? "当前已满足最小输入要求。" : "帧序列模式至少需要 2 张图片。"
    ];
  }
  if (model === "mast3r") {
    return [
      "MASt3R 更适合做静态多图重建对比，建议先用同一物体或小场景的 3 到 8 张图片测试。",
      "今晚先直接复用 DUSt3R 的标准参数，重点看匹配可视化和点云是否更稳。",
      fileCount >= 2 ? "当前已满足 MASt3R 的最小输入要求。" : "MASt3R 至少需要 2 张图片。"
    ];
  }
  return [
    "DUSt3R 最稳妥的起步方式是先用同一场景的 2 到 5 张图片。",
    "如果图片超过 6 张，建议把场景图改成 swin-5，避免 complete 配对过多。",
    fileCount >= 2 ? "当前已满足 DUSt3R 的最小输入要求。" : "DUSt3R 至少需要 2 张图片。"
  ];
}

function buildCreateReadiness(
  serviceReady: boolean,
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
  return [
    {
      label: "服务",
      value: serviceReady ? "Ready" : "Waiting",
      tone: serviceReady ? "ready" : "blocked"
    },
    {
      label: "模型",
      value: statusModelLabel(model),
      tone: "ready"
    },
    {
      label: "输入",
      value: inputReady ? `${fileCount} 个` : "待补",
      tone: inputReady ? "ready" : "partial"
    },
    {
      label: "来源",
      value: sourceTypeLabel(sourceType),
      tone: allowedSourceTypesForModel(model).includes(sourceType) ? "ready" : "blocked"
    }
  ];
}

function buildCreateLaunchHeadline(serviceReady: boolean, fileCount: number) {
  if (!serviceReady) {
    return "等待本地服务";
  }
  if (fileCount === 0) {
    return "先选择输入文件";
  }
  return "可以创建任务";
}

function buildCreateLaunchMessage(serviceReady: boolean, model: string, sourceType: string, fileCount: number) {
  if (!serviceReady) {
    return "本地服务就绪后才能提交，当前配置会保留在页面上。";
  }
  if (fileCount === 0) {
    return "先把文件放进 staging 区，再按当前模型和输入来源发起任务。";
  }
  return `${statusModelLabel(model)} / ${sourceTypeLabel(sourceType)} / ${fileCount} 个输入文件。`;
}

function buildCaptureChecklist(model: string, sourceType: string, fileCount: number) {
  if (model === "monst3r" && sourceType === "video") {
    return [
      {
        title: "视频先拍短一点",
        body: "先用 6 到 12 秒、主体稳定在画面中央的短视频起步，别一开始就上特别长的素材。"
      },
      {
        title: "保证明显视差",
        body: "尽量让相机绕主体缓慢移动，不要只做前后推拉，不然很容易塌成平面。"
      },
      {
        title: "当前上传检查",
        body: fileCount === 1 ? "视频数量正确，可以直接继续。" : "视频模式请只放 1 个视频文件。"
      }
    ];
  }

  if (model === "monst3r") {
    return [
      {
        title: "帧序列要连贯",
        body: "尽量用连续视角变化的帧，别混入跨度太大的图片。"
      },
      {
        title: "少量先验链路",
        body: "第一次先用 3 到 12 张连续帧验证流程，结果稳定后再补更长序列。"
      },
      {
        title: "当前上传检查",
        body: fileCount >= 2 ? "帧数已满足最小要求。" : "帧序列模式至少上传 2 张图片。"
      }
    ];
  }

  return [
    {
      title: "优先拍有纹理、有棱角的静态物体",
      body: "纸箱、工具箱、桌面物体都适合；玻璃、镜子、大白墙先尽量避开。"
    },
    {
      title: "多图要来自同一场景",
      body: "图片间要有重叠视角，主体尽量占画面的 40% 到 70%，别让背景比主体更抢眼。"
    },
    {
      title: "当前上传检查",
      body: fileCount >= 2 ? `当前已选 ${fileCount} 个文件，可以直接创建任务。` : "静态多图模式至少上传 2 张图片。"
    }
  ];
}

function buildAdvisorChecklist(advisorReady: boolean) {
  return [
    {
      title: "结果出来后第一时间评估",
      body: "任务完成、失败或取消时最值得点 AI 评估，它会结合参数、摘要、scene_meta 和日志给判断。"
    },
    {
      title: "别在纯运行中频繁点",
      body: "运行中的任务信息还不完整，更适合先看阶段卡片和最新日志，避免无效调用。"
    },
    {
      title: "直接拿来写汇报",
      body: advisorReady ? "AI 会给可直接复制的汇报话术，适合组会前快速整理表达。" : "先把 AI 接口配置好，才能启用自动诊断和汇报话术。"
    }
  ];
}

function buildSystemChecklist() {
  return [
    {
      title: "先建任务，再盯结果",
      body: "把上传和参数选择放在“文件与新建”，把状态、日志和产物放在“运行与结果”，别在首页来回翻。"
    },
    {
      title: "完成后再做 AI 总结",
      body: "一条任务结束后，先看核心产物，再去 AI 工作台生成诊断、下一步建议和汇报话术。"
    },
    {
      title: "本地服务异常先重启",
      body: "如果顶栏提示未连接，先在这里或顶栏重启本地服务，不要继续点任务按钮硬试。"
    }
  ];
}

function buildActionMessage(action: string, jobId: string) {
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
      return `任务 ${jobId} 的 AI 评估已更新。`;
    default:
      return `任务 ${jobId} 已更新。`;
  }
}

function isAdvisorSuggested(status: string) {
  return status === "finished" || status === "failed" || status === "cancelled";
}

function statusLabel(status: string) {
  switch (status) {
    case "running":
      return "运行中";
    case "finished":
      return "完成";
    case "failed":
      return "失败";
    case "cancelled":
      return "取消";
    case "draft":
      return "草稿";
    case "ready":
      return "就绪";
    default:
      return status;
  }
}

function statusModelLabel(model: string) {
  switch (model) {
    case "dust3r":
      return "DUSt3R";
    case "mast3r":
      return "MASt3R";
    case "monst3r":
      return "MonST3R";
    default:
      return model;
  }
}

function modelFamilyLabel(family: string) {
  const labels: Record<string, string> = {
    pairwise_pointmap: "Pairwise 点图",
    static_matching_reconstruction: "静态匹配重建",
    video_dynamic_reconstruction: "视频动态重建",
    memory_global_pointmap: "空间记忆",
    video_depth_consistency: "视频深度一致",
    large_image_collection: "长图集",
    streaming_state_reconstruction: "状态流式",
    general_visual_geometry: "通用视觉几何",
    stateful_linear_reconstruction: "线性状态",
    streaming_mapping: "流式建图"
  };
  return labels[family] ?? family.replace(/_/g, " ");
}

function runnerStatusLabel(status: string) {
  const labels: Record<string, string> = {
    baseline: "基座保留",
    validated_smoke: "Smoke 已过",
    validated_standard_sample: "标准样例已过",
    smoke_ready: "Smoke 已过",
    smoke_ready_attention_fallback: "Smoke 已过（需 attention fallback）",
    env_partial: "环境部分就绪",
    env_blocked_curope: "环境受 curope 阻塞",
    planned: "待接入",
    frontier_research: "前沿预研",
    integrated: "已接入"
  };
  return labels[status] ?? status.replace(/_/g, " ");
}

function formatDeploymentEnvSummary(payload: DeploymentStatusPayload) {
  const targets = ["mast3r", "monst3r", "spann3r", "align3r", "fast3r", "cut3r"];
  return payload.conda_envs
    .filter((item) => targets.includes(item.component))
    .map((item) => `${item.component}:${item.exists ? "OK" : "缺失"}${item.path ? ` (${item.path})` : ""}`)
    .join(" / ");
}

function formatDeploymentCacheStatus(payload: DeploymentStatusPayload | null) {
  if (!payload?.cache) {
    return "暂无缓存信息";
  }
  const state = payload.cache.state ?? payload.source ?? "unknown";
  const age = typeof payload.cache.age_seconds === "number" ? `${payload.cache.age_seconds.toFixed(1)}s` : "-";
  const fetched = payload.fetched_at ?? "-";
  return `${state} / age ${age} / fetched ${fetched}`;
}

function formatDeploymentDirectoryStatus(payload: DeploymentStatusPayload | null) {
  if (!payload?.directories?.length) {
    return "暂无目录状态";
  }
  return payload.directories
    .filter((item) => ["mast3r", "monst3r", "spann3r", "align3r", "fast3r", "cut3r"].includes(item.name))
    .map((item) => `${item.name}:${item.state}${item.readme_setup ? "" : "/README缺失"}`)
    .join(" / ");
}

function buildDeploymentComponentRows(payload: DeploymentStatusPayload | null) {
  if (!payload) {
    return [];
  }
  const targets = ["mast3r", "monst3r", "spann3r", "align3r", "fast3r", "cut3r"];
  return targets.map((component) => {
    const directory = payload.directories.find((item) => item.name === component);
    const env = payload.conda_envs.find((item) => item.component === component);
    const files = payload.known_files.filter((item) => item.component === component);
    const missingRequiredFiles = files.filter((item) => /required/i.test(item.need) && !item.exists).length;
    const checkpointCount = payload.checkpoints?.filter((item) => item.component === component).length ?? 0;
    const blocked = !directory?.exists || !env?.exists || missingRequiredFiles > 0;
    const tone = blocked ? "blocked" : checkpointCount > 0 ? "ready" : "partial";

    return {
      component,
      tone,
      directory: directory?.exists ? (directory.readme_setup ? "OK" : "README 待补") : "缺失",
      env: env?.exists ? "OK" : "缺失",
      files: files.length > 0 ? `${files.length - missingRequiredFiles}/${files.length}` : "未登记",
      checkpoints: checkpointCount > 0 ? `${checkpointCount} 个` : "待确认"
    };
  });
}

function modelDisplayName(value: string, catalog: ModelCatalogItem[]) {
  return catalog.find((item) => item.value === value)?.label ?? value.replace(/_/g, " ");
}

function formatModelList(values: string[], catalog: ModelCatalogItem[]) {
  if (values.length === 0) {
    return "暂无";
  }
  return values.map((value) => modelDisplayName(value, catalog)).join(" / ");
}

function formatCountMap(values: Record<string, number> | undefined, labeler: (value: string) => string) {
  if (!values || Object.keys(values).length === 0) {
    return "暂无";
  }
  return Object.entries(values)
    .map(([key, value]) => `${labeler(key)} ${value}`)
    .join(" / ");
}

function sampleStatusLabel(status: string) {
  const labels: Record<string, string> = {
    needs_selection: "待选样例"
  };
  if (/^seeded_from_job_/i.test(status)) {
    return "已有种子任务";
  }
  return labels[status] ?? status.replace(/_/g, " ");
}

function scoringCategoryLabel(category: string) {
  const labels: Record<string, string> = {
    engineering: "工程成本",
    result_quality: "结果质量",
    platform: "平台交付"
  };
  return labels[category] ?? category.replace(/_/g, " ");
}

function metricLabel(metric: string) {
  const labels: Record<string, string> = {
    setup_time: "环境准备",
    weight_download_difficulty: "权重获取",
    runtime_seconds: "运行耗时",
    peak_gpu_memory_mb: "峰值显存",
    runner_integration_difficulty: "Runner 接入",
    structure_completeness_1_to_5: "结构完整度",
    trajectory_stability_1_to_5: "轨迹稳定性",
    noise_level_1_to_5: "噪声水平",
    dynamic_handling_1_to_5: "动态处理",
    depth_temporal_consistency_1_to_5: "深度时序一致",
    presentation_usability_1_to_5: "展示可用性",
    noninteractive_runner: "非交互 Runner",
    status_json: "状态 JSON",
    scene_meta_json: "场景元数据",
    result_summary: "结果摘要",
    frontend_core_preview: "前端核心预览"
  };
  return labels[metric] ?? metric.replace(/_/g, " ");
}

function sourceTypeLabel(sourceType: string) {
  switch (sourceType) {
    case "images":
      return "图片";
    case "video":
      return "视频";
    case "frames":
      return "帧序列";
    default:
      return sourceType;
  }
}

function jobFilterLabel(filter: JobFilter) {
  switch (filter) {
    case "running":
      return "运行中";
    case "attention":
      return "待处理";
    case "finished":
      return "已完成";
    default:
      return "全部";
  }
}

function canDispatchJobStatus(status: string) {
  return status === "draft" || status === "ready" || status === "failed" || status === "cancelled";
}

function matchesJobQuery(item: JobListItem, normalizedQuery: string) {
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

function buildOverviewHeadline(job: JobListItem | null, runningCount: number, attentionCount: number) {
  if (!job) {
    return "先创建一条小任务，把整条链路确认跑通。";
  }
  if (job.job.status === "running") {
    return runningCount > 1 ? `当前有 ${runningCount} 条任务在跑，先盯住这一条。` : "当前有任务在跑，先观察这条的阶段推进。";
  }
  if (attentionCount > 0) {
    return "有任务需要人工处理，先看错误原因和最新日志。";
  }
  if (job.job.status === "finished") {
    return "已有结果回传，下一步优先检查核心成果和日志摘要。";
  }
  return "当前没有正在执行的任务，可以直接发起新的测试。";
}

function buildOverviewMessage(job: JobListItem | null, runningCount: number, attentionCount: number) {
  if (!job) {
    return "建议今晚先完成一条 DUSt3R 或 MonST3R 的正式样例，确认输入、远端执行、结果回传三段都稳定。";
  }
  if (job.job.status === "running") {
    return `${statusModelLabel(job.job.model)} 正在执行“${job.phase_display.label}”。如果进度长期不动，就先看详情页里的最新日志和阶段卡片。`;
  }
  if (attentionCount > 0) {
    return "失败或取消的任务已经单独归到“待处理”筛选里，先修复那里的阻塞，再继续批量测试。";
  }
  if (job.job.status === "finished") {
    return `当前焦点任务已经完成。建议先检查 ${statusModelLabel(job.job.model)} 的核心结果，再决定是否复制参数继续跑更多样例。`;
  }
  return runningCount > 0 ? "后台仍有其他任务在执行。" : "当前后端空闲，适合立刻发起下一条任务。";
}

function formatParamLabel(key: string) {
  const labels: Record<string, string> = {
    image_size: "图像尺寸",
    scene_graph: "场景图",
    niter: "对齐迭代",
    lr: "学习率",
    batch_size: "批大小",
    max_points: "最大点数",
    match_viz_count: "匹配线数量",
    fps: "抽帧 FPS",
    num_frames: "最大帧数",
    not_batchify: "省显存模式",
    real_time: "实时模式",
    window_wise: "窗口模式",
    window_size: "窗口大小",
    window_overlap_ratio: "窗口重叠率"
  };
  return labels[key] ?? key.replace(/_/g, " ");
}

function describeOutput(filename: string) {
  const lower = filename.toLowerCase();
  if (/dynamic_mask|enlarged_dynamic_mask/.test(lower)) {
    return "黑白动态掩膜，用来标记运动区域，不是最终重建结果";
  }
  if (/scene\.glb/.test(lower)) {
    return "MonST3R 三维场景，优先查看这个";
  }
  if (/pred_traj\.txt/.test(lower)) {
    return "MonST3R 相机轨迹，用来检查运动是否连续";
  }
  if (/pred_intrinsics\.txt/.test(lower)) {
    return "MonST3R 预测相机内参";
  }
  if (/^conf_\d+\.npy$/.test(lower) || /^init_conf_\d+\.npy$/.test(lower)) {
    return "MonST3R 置信数组，用于质量诊断";
  }
  if (/^frame_\d+\.npy$/.test(lower)) {
    return "MonST3R 每帧几何数组";
  }
  if (/frame_\d+\.png/.test(lower)) {
    return "彩色帧预览，可用来快速确认输入抽帧";
  }
  const suffix = filename.split(".").pop()?.toLowerCase();
  switch (suffix) {
    case "png":
    case "jpg":
    case "jpeg":
    case "webp":
      return "图像预览或匹配可视化";
    case "ply":
      return "点云模型，建议用 MeshLab 打开";
    case "glb":
    case "gltf":
      return "三维场景文件";
    case "txt":
      return "轨迹或相机文本";
    case "npy":
      return "数组产物";
    case "mp4":
    case "mov":
    case "avi":
    case "mkv":
    case "webm":
      return "视频产物";
    default:
      return "任务产物";
  }
}

function fileExtensionLabel(filename: string) {
  return filename.split(".").pop()?.toUpperCase() || "FILE";
}

function formatDuration(value: number | null) {
  if (!value || value <= 0) {
    return "-";
  }
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const seconds = value % 60;
  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds}s`;
  }
  return `${seconds}s`;
}

function formatDateTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  });
}

export default App;
