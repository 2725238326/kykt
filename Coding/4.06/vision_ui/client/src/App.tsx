import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type {
  AdvisorReport,
  BackendStatusPayload,
  BootstrapPayload,
  JobPayload,
  JobsListPayload,
  ResultSummary
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
type PreviewAsset = {
  url: string;
  name: string;
  kind: "image" | "video";
  note?: string;
};

function App() {
  const [bootstrap, setBootstrap] = useState<BootstrapPayload | null>(null);
  const [jobs, setJobs] = useState<JobsListPayload["jobs"]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<JobPayload | null>(null);
  const [backendStatus, setBackendStatus] = useState<BackendStatusPayload | null>(null);
  const [serviceState, setServiceState] = useState<ServiceState>("starting");
  const [serviceMessage, setServiceMessage] = useState("正在准备本地服务...");
  const [submitting, setSubmitting] = useState(false);
  const [actionKey, setActionKey] = useState<string | null>(null);
  const [jobFilter, setJobFilter] = useState<JobFilter>("all");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [infoMessage, setInfoMessage] = useState<string | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [recoveringService, setRecoveringService] = useState(false);
  const [previewAsset, setPreviewAsset] = useState<PreviewAsset | null>(null);
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
  const serviceReady = serviceState === "ready";
  const isImageCollectionModel = formState.model === "dust3r" || formState.model === "mast3r";
  const isMonst3r = formState.model === "monst3r";
  const selectedModel = useMemo(
    () => bootstrapData.models.find((item) => item.value === formState.model),
    [bootstrapData.models, formState.model]
  );
  const selectedListJob = useMemo(
    () => (selectedJobId ? jobs.find((item) => item.job.job_id === selectedJobId) ?? null : null),
    [jobs, selectedJobId]
  );
  const runningJobs = useMemo(() => jobs.filter((item) => item.job.status === "running"), [jobs]);
  const attentionJobs = useMemo(
    () => jobs.filter((item) => item.job.status === "failed" || item.job.status === "cancelled"),
    [jobs]
  );
  const finishedJobs = useMemo(() => jobs.filter((item) => item.job.status === "finished"), [jobs]);
  const focusJob = useMemo(
    () => selectedListJob ?? runningJobs[0] ?? attentionJobs[0] ?? jobs[0] ?? null,
    [selectedListJob, runningJobs, attentionJobs, jobs]
  );
  const filteredJobs = useMemo(() => {
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
  }, [attentionJobs, finishedJobs, jobFilter, jobs, runningJobs]);
  const createGuidance = useMemo(
    () => buildCreateGuidance(formState.model, formState.source_type, files.length),
    [files.length, formState.model, formState.source_type]
  );
  const summary = bootstrap?.summary ?? {
    total: jobs.length,
    running: jobs.filter((item) => item.job.status === "running").length,
    finished: jobs.filter((item) => item.job.status === "finished").length,
    failed: jobs.filter((item) => item.job.status === "failed").length,
    cancelled: jobs.filter((item) => item.job.status === "cancelled").length
  };
  const runningSelectedJob = selectedJob?.job.status === "running";
  const canDispatchSelectedJob = selectedJob
    ? selectedJob.job.status === "draft" ||
      selectedJob.job.status === "ready" ||
      selectedJob.job.status === "failed" ||
      selectedJob.job.status === "cancelled"
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
          const payload = await fetchJson<BootstrapPayload>("/api/bootstrap");
          setBootstrap(payload);
          setServiceState("ready");
          setServiceMessage("本地服务已就绪");
          await loadJobs(false);
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
    const timer = window.setInterval(() => void loadJobs(false), 4000);
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
    const timer = window.setInterval(() => void loadJobDetail(selectedJobId, false), 4000);
    return () => window.clearInterval(timer);
  }, [selectedJobId, serviceReady]);

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

      const payload = await fetchJson<BootstrapPayload>("/api/bootstrap");
      setBootstrap(payload);
      setServiceState("ready");
      setServiceMessage("本地服务已恢复并重新连通。");
      setErrorMessage(null);
      if (announce) {
        setInfoMessage(mode === "restart" ? "本地服务已重启。" : "本地服务已恢复。");
      }
      await loadJobs(false);
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

        {infoMessage ? <MessageBanner kind="info" message={infoMessage} /> : null}
        {errorMessage ? <MessageBanner kind="error" message={errorMessage} /> : null}

        <section className="overview-grid">
          <article className="panel overview-hero-panel">
            <PanelTitle eyebrow="任务总览" title={focusJob ? `焦点任务 ${focusJob.job.job_id}` : "准备开始今晚测试"} />
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
                  <button
                    className="ghost-button small"
                    onClick={() => setSelectedJobId(focusJob.job.job_id)}
                    type="button"
                  >
                    查看详情
                  </button>
                </div>
              </div>
            ) : (
              <div className="empty-state large">
                还没有任务。建议先从 2 张图片的 DUSt3R 或 1 段短视频的 MonST3R 开始，今晚先把一条样例完整跑通。
              </div>
            )}
          </article>

          <aside className="panel overview-side-panel">
            <PanelTitle eyebrow="调度状态" title="当前面板" />
            <div className="kpi-grid">
              <MiniStat label="总任务" value={summary.total} />
              <MiniStat label="运行中" value={summary.running} />
              <MiniStat label="待处理" value={attentionJobs.length} />
              <MiniStat label="已完成" value={summary.finished} />
            </div>
            <div className={`overview-callout ${focusJob?.job.status ?? "neutral"}`}>
              <span className="mini-label">当前建议</span>
              <strong>{buildOverviewHeadline(focusJob, runningJobs.length, attentionJobs.length)}</strong>
              <p>{buildOverviewMessage(focusJob, runningJobs.length, attentionJobs.length)}</p>
            </div>
          </aside>
        </section>

        <section className="layout-grid">
          <article className="panel create-panel">
            <PanelTitle eyebrow="新建任务" title="选择模型和输入" />
            <form className="form-stack" onSubmit={handleCreateJob}>
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

              <article className="create-guidance">
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
              </article>

              <label className="dropzone">
                <input
                  type="file"
                  multiple
                  onChange={(event) => setFiles(Array.from(event.target.files ?? []))}
                />
                <span>点击选择文件</span>
                <small>{inputHint(formState.model, formState.source_type)}</small>
              </label>

              {files.length > 0 ? (
                <div className="file-list">
                  {files.map((file) => (
                    <button
                      key={`${file.name}-${file.size}`}
                      className="file-chip"
                      type="button"
                      onClick={() => removePendingFile(file.name, file.size)}
                    >
                      <span>{file.name}</span>
                      <strong>移除</strong>
                    </button>
                  ))}
                </div>
              ) : null}

              <details className="advanced-panel">
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

              <button className="primary-button" disabled={!serviceReady || submitting} type="submit">
                {submitting ? "创建中..." : `创建 ${selectedModel?.label ?? "模型"} 任务`}
              </button>
            </form>
          </article>

          <aside className="panel side-panel">
            <PanelTitle eyebrow="最近任务" title={`${summary.total} 个任务`} />
            <div className="mini-stats">
              <MiniStat label="总任务" value={summary.total} />
              <MiniStat label="运行" value={summary.running} />
              <MiniStat label="待处理" value={attentionJobs.length} />
              <MiniStat label="完成" value={summary.finished} />
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
                  {jobFilter === "all"
                    ? "暂无任务。先在左侧选择输入文件，创建第一条任务。"
                    : `当前筛选下没有${jobFilterLabel(jobFilter)}任务。`}
                </div>
              )}
            </div>
          </aside>
        </section>

        <section className="panel detail-panel">
          <PanelTitle eyebrow="任务详情" title={selectedJob?.job.job_id ?? "尚未选择任务"} />
          {selectedJob ? (
            <JobDetail
              selectedJob={selectedJob}
              actionKey={actionKey}
              canDispatch={canDispatchSelectedJob}
              running={Boolean(runningSelectedJob)}
              assetUrl={assetUrl}
              onAction={postJobAction}
              onOpenOutput={openOutput}
              onPreviewAsset={openPreviewAsset}
            />
          ) : (
            <div className="empty-state large">
              这里会显示任务进度、输入预览、输出结果和日志。页面先保持干净，不再一打开就塞满信息。
            </div>
          )}
        </section>
      </main>

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

function JobDetail(props: {
  selectedJob: JobPayload;
  actionKey: string | null;
  canDispatch: boolean;
  running: boolean;
  assetUrl: (path: string) => string;
  onAction: (path: string, key: string) => Promise<void>;
  onOpenOutput: (relativePath: string) => Promise<void>;
  onPreviewAsset: (asset: PreviewAsset) => void;
}) {
  const job = props.selectedJob.job;
  const summary = props.selectedJob.result_summary;
  const latestLogLine = getLatestLogLine(props.selectedJob.logs);
  const outputSections = buildOutputSections(props.selectedJob.outputs, job.model);
  const progress = props.selectedJob.phase_display;

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
        <button
          disabled={props.actionKey === "advisor"}
          onClick={() => props.onAction(`/api/jobs/${job.job_id}/advisor/evaluate`, "advisor")}
          type="button"
        >
          AI评估
        </button>
      </div>

      <div className="meta-grid">
        <MetaCard label="创建时间" value={formatDateTime(job.created_at)} />
        <MetaCard label="输入数量" value={String(summary?.inputs?.count ?? job.input_items.length ?? 0)} />
        <MetaCard label="回传产物" value={String(summary?.artifacts?.length ?? props.selectedJob.outputs.length)} />
        <MetaCard label="最新日志" value={latestLogLine || "尚无有效日志"} compact />
      </div>

      {job.error_message ? (
        <article className="soft-panel error-panel">
          <h4>错误原因</h4>
          <p>{job.error_message}</p>
        </article>
      ) : null}

      <div className="result-grid">
        <article className="soft-panel">
          <h4>结果摘要</h4>
          <SummaryPanel summary={props.selectedJob.result_summary} />
        </article>
        <article className="soft-panel">
          <h4>AI 评估</h4>
          <AdvisorPanel report={props.selectedJob.advisor_report ?? null} />
        </article>
        <article className="soft-panel">
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
      </div>

      <article className="soft-panel">
        <div className="section-head">
          <div>
            <h4>输出结果</h4>
            <p>按用途分组展示，不再把 MonST3R / DUSt3R 产物全都摊成一排。</p>
          </div>
          <span className="section-pill">{props.selectedJob.outputs.length} 个文件</span>
        </div>
        {outputSections.length > 0 ? (
          <div className="output-section-list">
            {outputSections.map((section) => (
              <details className={`output-section ${section.accent}`} key={section.key} open={section.defaultOpen}>
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

      <article className="soft-panel">
        <div className="section-head">
          <div>
            <h4>日志</h4>
            <p>优先看最新一条有效进展，再往下翻完整日志尾部。</p>
          </div>
          <span className="section-pill">{props.selectedJob.logs.length} 份</span>
        </div>
        {latestLogLine ? <div className="latest-log-banner">{latestLogLine}</div> : null}
        {props.selectedJob.logs.length > 0 ? (
          <div className="log-list">
            {props.selectedJob.logs.map((log) => (
              <div className="log-card" key={log.relative_path}>
                <strong>{log.name}</strong>
                <pre>{log.tail || "暂无日志。"}</pre>
              </div>
            ))}
          </div>
        ) : (
          <span className="muted-text">还没有日志。</span>
        )}
      </article>
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
  const sceneMeta = props.summary.scene_meta ?? {};
  const sceneStats = [
    typeof sceneMeta["artifact_count"] === "number" ? { label: "远端整理", value: String(sceneMeta["artifact_count"]) } : null,
    typeof sceneMeta["glb_count"] === "number" ? { label: "GLB", value: String(sceneMeta["glb_count"]) } : null,
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
