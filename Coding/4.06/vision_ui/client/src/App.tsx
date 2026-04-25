import { FormEvent, KeyboardEvent as ReactKeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
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
  SamplesPayload
} from "./types";
import {
  API_BASE,
  DEFAULT_BOOTSTRAP,
  defaultDust3rParams,
  defaultFast3rParams,
  defaultMonst3rParams,
  dust3rParamChoices,
  fast3rParamChoices,
  monst3rParamChoices,
  presetDescriptors
} from "./appConfig";
import type { ParamChoice, PresetKey } from "./appConfig";
import {
  backendStatusText,
  delay,
  describeOutput,
  fileExtensionLabel,
  formatDateTime,
  formatParamLabel,
  friendlyError,
  jobFilterLabel,
  modelDisplayName,
  runnerStatusLabel,
  serviceStatusLabel,
  sourceTypeLabel,
  statusLabel,
  statusModelLabel
} from "./displayHelpers";
import type { ModelCatalogItem } from "./displayHelpers";
import {
  buildDeploymentComponentRows,
  buildModelActionRows,
  formatDeploymentCacheStatus,
  formatDeploymentDirectoryStatus,
  formatDeploymentEnvSummary
} from "./deploymentHelpers";
import {
  formatFileSize,
  isImageLikeFile,
  isVideoLikeFile,
  pendingFileRoleLabel
} from "./fileHelpers";
import {
  allowedSourceTypesForModel,
  batchActionLabel,
  buildActionMessage,
  buildAdvisorChecklist,
  buildCaptureChecklist,
  buildCreateGuidance,
  buildCreateLaunchHeadline,
  buildCreateLaunchMessage,
  buildCreateReadiness,
  buildDust3rPreset,
  buildFast3rPreset,
  buildModelLaunchBlocker,
  buildMonst3rPreset,
  buildOverviewHeadline,
  buildOverviewMessage,
  buildParamPanelIntro,
  buildSystemChecklist,
  canDispatchJobStatus,
  fallbackParamFamilyForModel,
  fallbackSourceTypesForModel,
  getCreateParamMode,
  getParamDefaultsForMode,
  inputHint,
  isAdvisorSuggested,
  isParamFieldKey,
  matchesJobQuery,
  presetLabel
} from "./workflowHelpers";
import type {
  BatchJobAction,
  CreateParamMode,
  FormState,
  JobListItem
} from "./workflowHelpers";
import {
  FilterPill,
  MessageBanner,
  MiniStat,
  ModelSemanticChips,
  PanelTitle,
  StatusBadge,
  SummaryStat
} from "./uiPrimitives";
import { ModelRoadmapPanel } from "./ModelRoadmapPanel";
import { SummaryPanel } from "./SummaryPanel";
import { SampleMatrixPanel } from "./SampleMatrixPanel";
import { EvaluationPanel } from "./EvaluationPanel";

type ServiceState = "starting" | "ready" | "degraded";
type JobFilter = "all" | "running" | "attention" | "finished";
type WorkspaceTab = "overview" | "create" | "jobs" | "advisor" | "system";
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

const workspaceTabs: Array<{ key: WorkspaceTab; label: string; note: string }> = [
  { key: "overview", label: "工作台", note: "全局状态与焦点任务" },
  { key: "create", label: "文件与新建", note: "整理输入并创建任务" },
  { key: "jobs", label: "运行与结果", note: "筛选任务、跟进状态、查看产物" },
  { key: "advisor", label: "辅助评估", note: "自动评估草稿" },
  { key: "system", label: "帮助与系统", note: "服务与远端部署" }
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
  const [activePresets, setActivePresets] = useState<Record<string, PresetKey | null>>({
    dust3r: "standard",
    mast3r: "standard",
    monst3r: "standard",
    spann3r: null,
    fast3r: "standard"
  });
  const recoveryInFlightRef = useRef(false);
  const jobSearchInputRef = useRef<HTMLInputElement | null>(null);
  const [formState, setFormState] = useState<FormState>({
    model: "dust3r",
    source_type: "images",
    notes: "",
    ...defaultDust3rParams,
    ...defaultMonst3rParams,
    ...defaultFast3rParams
  });

  const bootstrapData = bootstrap ?? DEFAULT_BOOTSTRAP;
  const advisorState: AdvisorStatus = bootstrapData.advisor ?? {
    enabled: false,
    configured: false,
    base_url: "",
    model: "",
    has_api_key: false,
    message: "辅助评估尚未配置。"
  };
  const serviceReady = serviceState === "ready";
  const advisorReady = advisorState.enabled && advisorState.configured;
  const modelCatalog = useMemo<ModelCatalogItem[]>(
    () =>
      samplesPayload?.model_catalog ??
      bootstrapData.model_catalog ??
      bootstrapData.models.map((item) => ({
        value: item.value,
        label: item.label,
        description: item.description,
        family: item.family ?? "integrated",
        param_family: item.param_family ?? fallbackParamFamilyForModel(item.value),
        source_types: item.source_types ?? fallbackSourceTypesForModel(item.value),
        runner_status: item.runner_status ?? "integrated",
        research_priority: item.research_priority ?? 0,
        active_track: item.active_track ?? true,
        runnable: item.runnable ?? true,
        launch_blocker: item.launch_blocker ?? null
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
  const runnableModelCatalog = useMemo(() => modelCatalog.filter((item) => item.runnable), [modelCatalog]);
  const catalogOnlyModelCatalog = useMemo(() => modelCatalog.filter((item) => !item.runnable), [modelCatalog]);
  const selectedModelCatalog = useMemo(
    () => modelCatalog.find((item) => item.value === formState.model) ?? null,
    [formState.model, modelCatalog]
  );
  const selectedModel = useMemo(
    () => selectedModelCatalog ?? bootstrapData.models.find((item) => item.value === formState.model) ?? null,
    [bootstrapData.models, formState.model, selectedModelCatalog]
  );
  const createParamMode = useMemo(
    () => getCreateParamMode(formState.model, modelCatalog),
    [formState.model, modelCatalog]
  );
  const selectedModelSourceTypes = useMemo(
    () => allowedSourceTypesForModel(formState.model, modelCatalog),
    [formState.model, modelCatalog]
  );
  const selectedModelLaunchBlocker = useMemo(
    () => buildModelLaunchBlocker(selectedModelCatalog),
    [selectedModelCatalog]
  );
  const activePreset = activePresets[formState.model] ?? null;
  const supportsPresets = createParamMode !== "spann3r_sequence" && createParamMode !== "catalog";
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
  const batchDispatchJobs = useMemo(
    () => filteredJobs.filter((item) => canDispatchJobStatus(item.job.status)),
    [filteredJobs]
  );
  const batchRetryJobs = useMemo(
    () => filteredJobs.filter((item) => item.job.status === "failed" || item.job.status === "cancelled"),
    [filteredJobs]
  );
  const batchCancelJobs = useMemo(
    () => filteredJobs.filter((item) => item.job.status === "running"),
    [filteredJobs]
  );
  const batchActionBusy = actionKey?.startsWith("batch:") ?? false;
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
    () =>
      buildCreateReadiness(
        serviceReady,
        selectedModelCatalog,
        selectedModelSourceTypes,
        formState.model,
        formState.source_type,
        files.length,
        pendingImageCount,
        pendingVideoCount
      ),
    [files.length, formState.model, formState.source_type, pendingImageCount, pendingVideoCount, selectedModelCatalog, selectedModelSourceTypes, serviceReady]
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
    if (activeWorkspace !== "jobs" || filteredJobs.length === 0) {
      return;
    }
    if (!selectedJobId || selectedFilteredIndex < 0) {
      setSelectedJobId(filteredJobs[0].job.job_id);
    }
  }, [activeWorkspace, filteredJobs, selectedFilteredIndex, selectedJobId]);

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
    window.addEventListener("keydown", handleGlobalJobHotkeys);
    return () => window.removeEventListener("keydown", handleGlobalJobHotkeys);
  }, [activeWorkspace, filteredJobs.length, selectedFilteredIndex]);

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
      setInfoMessage(payload.configured ? "辅助评估配置已保存。" : "评估配置已保存，但尚未可用。");
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
      setActivePresets((current) => ({ ...current, [formState.model]: null }));
    }
  }

  function updateModel(value: string) {
    setFormState((current) => {
      const nextSourceTypes = allowedSourceTypesForModel(value, modelCatalog);
      const nextSourceType = nextSourceTypes.includes(current.source_type) ? current.source_type : nextSourceTypes[0] ?? "images";
      const nextParamMode = getCreateParamMode(value, modelCatalog);
      const baseState = {
        ...current,
        model: value,
        source_type: nextSourceType
      };
      switch (nextParamMode) {
        case "video_sequence":
          return {
            ...baseState,
            ...defaultMonst3rParams
          };
        case "fast3r_collection":
          return {
            ...baseState,
            ...defaultFast3rParams
          };
        case "spann3r_sequence":
        case "catalog":
          return baseState;
        case "image_collection":
        default:
          return {
            ...baseState,
            ...defaultDust3rParams
          };
      }
    });
    const nextParamMode = getCreateParamMode(value, modelCatalog);
    setActivePresets((current) => ({
      ...current,
      [value]: nextParamMode === "spann3r_sequence" || nextParamMode === "catalog" ? null : "standard"
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

  function focusJobSearch() {
    jobSearchInputRef.current?.focus();
    jobSearchInputRef.current?.select();
  }

  function handleGlobalJobHotkeys(event: KeyboardEvent) {
    if (activeWorkspace !== "jobs") {
      return;
    }
    const target = event.target as HTMLElement | null;
    const isTypingTarget =
      target instanceof HTMLInputElement ||
      target instanceof HTMLTextAreaElement ||
      target instanceof HTMLSelectElement ||
      target?.isContentEditable;
    if ((event.key === "j" || event.key === "k") && !isTypingTarget) {
      event.preventDefault();
      moveSelectedJob(event.key === "j" ? 1 : -1);
      return;
    }
    if (event.key === "/" && !isTypingTarget) {
      event.preventDefault();
      focusJobSearch();
    }
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
      switch (getCreateParamMode(current.model, modelCatalog)) {
        case "video_sequence":
          return {
            ...current,
            ...buildMonst3rPreset(preset)
          };
        case "fast3r_collection":
          return {
            ...current,
            ...buildFast3rPreset(preset)
          };
        case "spann3r_sequence":
        case "catalog":
          return current;
        case "image_collection":
        default:
          return {
            ...current,
            ...buildDust3rPreset(preset, files.length)
          };
      }
    });
    setActivePresets((current) => ({
      ...current,
      [formState.model]: preset
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
    if (selectedModelLaunchBlocker) {
      setErrorMessage(selectedModelLaunchBlocker);
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

    const paramDefaults = getParamDefaultsForMode(createParamMode);
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
    if (!selectedModelSourceTypes.includes(formState.source_type)) {
      return `${statusModelLabel(formState.model)} 当前不支持 ${sourceTypeLabel(formState.source_type)} 输入。`;
    }
    if (formState.source_type === "video") {
      if (files.length !== 1 || !files.every((file) => isVideoLikeFile(file))) {
        return `${statusModelLabel(formState.model)} 视频模式请上传 1 个视频文件。`;
      }
      return null;
    }
    if (!files.every((file) => isImageLikeFile(file))) {
      return `${statusModelLabel(formState.model)} 当前输入类型需要图片或帧序列文件，不要混入视频。`;
    }
    if (files.length < 2) {
      return formState.source_type === "frames"
        ? `${statusModelLabel(formState.model)} 帧序列模式至少上传 2 张图片。`
        : `${statusModelLabel(formState.model)} 至少需要两张输入图片。`;
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

  async function postJobsBatchAction(targetJobs: JobListItem[], action: BatchJobAction) {
    if (targetJobs.length === 0) {
      return;
    }

    const batchKey = `batch:${action}`;
    const label = batchActionLabel(action);
    setActionKey(batchKey);
    setErrorMessage(null);
    setInfoMessage(null);

    let completed = 0;
    let lastPayload: JobPayload | null = null;
    const failures: string[] = [];

    try {
      for (const item of targetJobs) {
        try {
          const payload = await fetchJson<JobPayload>(`/api/jobs/${item.job.job_id}/${action}`, { method: "POST" });
          completed += 1;
          lastPayload = payload;
        } catch (error) {
          failures.push(`${item.job.job_id}: ${friendlyError(error, "操作失败")}`);
        }
      }

      const focusJobId = lastPayload?.job.job_id ?? selectedJobId;
      if (lastPayload) {
        setSelectedJob(lastPayload);
        setSelectedJobId(lastPayload.job.job_id);
      }
      await loadJobs(false);
      if (focusJobId) {
        await loadJobDetail(focusJobId, false);
      }

      if (failures.length > 0) {
        setErrorMessage(`批量${label}完成 ${completed}/${targetJobs.length} 个，失败 ${failures.length} 个。${failures.slice(0, 2).join("；")}`);
      } else {
        setInfoMessage(`已批量${label} ${completed} 个任务。`);
      }
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
              <span className="mini-label">辅助评估</span>
              <strong>
                {advisorReady
                  ? `已就绪 · ${advisorState.model || "已配置模型"}`
                  : advisorState.enabled
                    ? "已启用但尚未完整配置"
                    : "尚未启用"}
              </strong>
            </div>
            <div className="service-card-copy">
              <p>{advisorReady ? "已可生成自动评估草稿。" : advisorState.message}</p>
              <div className="service-card-actions">
                <button className="ghost-button small" onClick={() => openWorkspace("advisor")} type="button">
                  打开评估
                </button>
                <button className="ghost-button small" onClick={() => void openAdvisorSettings()} disabled={advisorConfigLoading} type="button">
                  {advisorConfigLoading ? "读取中..." : "配置"}
                </button>
                {advisorReady && activeJob && isAdvisorSuggested(activeJob.job.status) ? (
                  <button
                    className="ghost-button small"
                    onClick={() => void postJobAction(`/api/jobs/${activeJob.job.job_id}/advisor/evaluate`, "advisor")}
                    disabled={actionKey === "advisor"}
                    type="button"
                  >
                    生成草稿
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
                        <optgroup label="可创建模型">
                          {runnableModelCatalog.map((item) => (
                            <option key={item.value} value={item.value}>
                              {item.label}
                            </option>
                          ))}
                        </optgroup>
                        {catalogOnlyModelCatalog.length > 0 ? (
                          <optgroup label="目录模型（暂不可创建）">
                            {catalogOnlyModelCatalog.map((item) => (
                              <option key={item.value} value={item.value} disabled>
                                {item.label} · {runnerStatusLabel(item.runner_status)}
                              </option>
                            ))}
                          </optgroup>
                        ) : null}
                      </select>
                    </label>
                    <label className="field">
                      <span>输入类型</span>
                      <select
                        value={formState.source_type}
                        onChange={(event) => updateFormField("source_type", event.target.value)}
                      >
                        {bootstrapData.source_types
                          .filter((item) => selectedModelSourceTypes.includes(item.value))
                          .map((item) => (
                            <option key={item.value} value={item.value}>
                              {item.label}
                            </option>
                          ))}
                      </select>
                    </label>
                  </div>

                  <div className="create-model-context">
                    <div className="create-model-context-head">
                      <strong>{selectedModel?.description ?? "模型说明"}</strong>
                      {selectedModelCatalog ? (
                        <span className={`create-model-badge ${selectedModelCatalog.runnable ? "runnable" : "catalog"}`}>
                          {selectedModelCatalog.runnable ? "可创建" : "目录模型"}
                        </span>
                      ) : null}
                    </div>
                    <ModelSemanticChips
                      catalog={modelCatalog}
                      className="create-model-facts"
                      model={formState.model}
                      showParamFamily
                    />
                    {selectedModelLaunchBlocker ? (
                      <p className="create-model-note blocked">{selectedModelLaunchBlocker}</p>
                    ) : null}
                    {catalogOnlyModelCatalog.length > 0 ? (
                      <div className="create-model-blockers" aria-label="目录模型阻塞原因">
                        {catalogOnlyModelCatalog.map((item) => (
                          <article key={item.value}>
                            <div>
                              <strong>{item.label}</strong>
                              <span>{runnerStatusLabel(item.runner_status)}</span>
                            </div>
                            <p>{buildModelLaunchBlocker(item)}</p>
                          </article>
                        ))}
                      </div>
                    ) : null}
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
                    <div className="empty-state">暂无待上传文件。</div>
                  )}
                </article>

                <article className="create-block create-guidance-block">
                  <div className="create-guidance">
                    <div className="guide-head">
                      <div>
                        <strong>输入状态</strong>
                        <p>{selectedModel?.description ?? "模型说明"}</p>
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
                      {buildParamPanelIntro(createParamMode)}
                    </div>
                    {supportsPresets ? (
                      <div className="preset-strip">
                        <div className="preset-strip-head">
                          <strong>一键预设</strong>
                          <span>{activePreset ? `当前：${presetLabel(activePreset)}` : "当前：已手动调整"}</span>
                        </div>
                        <div className="preset-pills">
                          {presetDescriptors.map((preset) => (
                            <button
                              key={preset.key}
                              className={`preset-pill ${activePreset === preset.key ? "active" : ""}`}
                              type="button"
                              onClick={() => applyPreset(preset.key)}
                            >
                              <strong>{preset.label}</strong>
                            </button>
                          ))}
                        </div>
                      </div>
                    ) : null}
                    {createParamMode === "image_collection" ? (
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
                    ) : createParamMode === "video_sequence" ? (
                      <Monst3rParams formState={formState} updateFormField={updateFormField} />
                    ) : createParamMode === "fast3r_collection" ? (
                      <Fast3rParams formState={formState} updateFormField={updateFormField} />
                    ) : createParamMode === "spann3r_sequence" ? (
                      <Spann3rParams />
                    ) : (
                      <CatalogOnlyParams blocker={selectedModelLaunchBlocker} />
                    )}
                  </details>
                </article>
              </section>

              <div className="create-submit-dock">
                <div>
                  <span className="mini-label">Launch</span>
                  <strong>{buildCreateLaunchHeadline(serviceReady, files.length, selectedModelLaunchBlocker)}</strong>
                  <p>{buildCreateLaunchMessage(serviceReady, formState.model, formState.source_type, files.length, selectedModelLaunchBlocker)}</p>
                </div>
                <button className="primary-button" disabled={!serviceReady || submitting || Boolean(selectedModelLaunchBlocker)} type="submit">
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
                    ref={jobSearchInputRef}
                    value={jobQuery}
                    onChange={(event) => setJobQuery(event.target.value)}
                    onKeyDown={(event: ReactKeyboardEvent<HTMLInputElement>) => {
                      if (event.key === "ArrowDown") {
                        event.preventDefault();
                        moveSelectedJob(1);
                      }
                      if (event.key === "ArrowUp") {
                        event.preventDefault();
                        moveSelectedJob(-1);
                      }
                    }}
                    placeholder="任务ID / 模型 / 状态 / 输入类型"
                  />
                  <p className="field-note jobs-shortcut-note">快捷键：/ 聚焦检索，J / K 切换选中任务。</p>
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
                        ? `${modelDisplayName(selectedListJob.job.model, modelCatalog)} · ${statusLabel(selectedListJob.job.status)}`
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
                    <button className="ghost-button small" type="button" onClick={focusJobSearch}>
                      检索
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
                <div className="jobs-batch-strip" aria-label="筛选范围批量操作">
                  <div className="jobs-batch-summary">
                    <span className="mini-label">批量操作</span>
                    <strong>{filteredJobs.length > 0 ? `筛选范围 ${filteredJobs.length} 条任务` : "当前筛选为空"}</strong>
                    <p>
                      可运行 {batchDispatchJobs.length} · 待处理可重试 {batchRetryJobs.length} · 运行中可取消 {batchCancelJobs.length}
                    </p>
                  </div>
                  <div className="jobs-batch-actions">
                    <button
                      className="ghost-button small"
                      type="button"
                      disabled={batchDispatchJobs.length === 0 || Boolean(actionKey)}
                      onClick={() => void postJobsBatchAction(batchDispatchJobs, "dispatch")}
                    >
                      {actionKey === "batch:dispatch" ? "批量运行中..." : `运行 ${batchDispatchJobs.length}`}
                    </button>
                    <button
                      className="ghost-button small"
                      type="button"
                      disabled={batchRetryJobs.length === 0 || Boolean(actionKey)}
                      onClick={() => void postJobsBatchAction(batchRetryJobs, "retry")}
                    >
                      {actionKey === "batch:retry" ? "批量重试中..." : `重试 ${batchRetryJobs.length}`}
                    </button>
                    <button
                      className="ghost-button small danger"
                      type="button"
                      disabled={batchCancelJobs.length === 0 || Boolean(actionKey)}
                      onClick={() => void postJobsBatchAction(batchCancelJobs, "cancel")}
                    >
                      {actionKey === "batch:cancel" ? "批量取消中..." : `取消 ${batchCancelJobs.length}`}
                    </button>
                  </div>
                </div>
                {selectedListJob ? (
                  <div className="jobs-inline-actions">
                    <button
                      className="ghost-button small"
                      type="button"
                      disabled={!canDispatchListJob || actionKey === "dispatch" || batchActionBusy}
                      onClick={() => void postJobAction(`/api/jobs/${selectedListJob.job.job_id}/dispatch`, "dispatch")}
                    >
                      {actionKey === "dispatch" ? "运行中..." : "运行"}
                    </button>
                    <button
                      className="ghost-button small"
                      type="button"
                      disabled={!canRetryListJob || actionKey === "retry" || batchActionBusy}
                      onClick={() => void postJobAction(`/api/jobs/${selectedListJob.job.job_id}/retry`, "retry")}
                    >
                      {actionKey === "retry" ? "重试中..." : "重试"}
                    </button>
                    <button
                      className="ghost-button small danger"
                      type="button"
                      disabled={!canCancelListJob || actionKey === "cancel" || batchActionBusy}
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
                          {modelDisplayName(item.job.model, modelCatalog)} · {statusLabel(item.job.status)}
                        </span>
                        <ModelSemanticChips
                          catalog={modelCatalog}
                          className="job-model-semantics"
                          compact
                          model={item.job.model}
                        />
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
                  modelCatalog={modelCatalog}
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
              <PanelTitle eyebrow="Evaluation" title={selectedJob?.job.job_id ?? "先选中一条任务"} />
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
              <PanelTitle eyebrow="Evaluation" title="当前范围" />
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
                <span className="mini-label">评估配置</span>
                <strong>OpenAI 兼容接口</strong>
                <p>API Key 留空则保持不变。</p>
              </div>
              <button className="ghost-button small" onClick={() => setAdvisorModalOpen(false)} type="button">
                关闭
              </button>
            </div>

            <form className="form-stack settings-form" onSubmit={saveAdvisorSettings}>
              <label className="field">
                <span>启用辅助评估</span>
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
          <PanelTitle eyebrow="Focus" title={focusJob ? focusJob.job.job_id : "准备开始测试"} />
          {focusJob ? (
            <div className={`focus-card ${focusJob.job.status}`}>
              <div className="focus-main">
                <div className="focus-copy">
                  <div className="hero-badges">
                    <StatusBadge state={focusJob.job.status} label={statusLabel(focusJob.job.status)} />
                    <span className="hero-tag">{modelDisplayName(focusJob.job.model, props.modelCatalog)}</span>
                    <span className="hero-tag">{sourceTypeLabel(focusJob.job.source_type)}</span>
                  </div>
                  <ModelSemanticChips
                    catalog={props.modelCatalog}
                    className="detail-model-semantics"
                    compact
                    model={focusJob.job.model}
                  />
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
            <div className="empty-state large">还没有任务。先创建一条样例任务。</div>
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
            <span className="mini-label">当前状态</span>
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
          <PanelTitle eyebrow="Workflow" title="工作流" />
          <div className="quick-action-grid">
            <button className="quick-action-card" onClick={() => props.openWorkspace("create")} type="button">
              <strong>新建任务</strong>
              <p>模型、来源、文件。</p>
            </button>
            <button className="quick-action-card" onClick={() => props.openWorkspace("jobs")} type="button">
              <strong>运行与结果</strong>
              <p>任务、产物、日志。</p>
            </button>
            <button className="quick-action-card" onClick={() => props.openWorkspace("advisor")} type="button">
              <strong>辅助评估</strong>
              <p>自动评估草稿。</p>
            </button>
            <button className="quick-action-card" onClick={() => props.openWorkspace("system")} type="button">
              <strong>系统与部署</strong>
              <p>服务与远端状态。</p>
            </button>
          </div>
        </article>

        <article className="panel advisor-overview-panel">
          <PanelTitle eyebrow="Evaluation" title="辅助评估" />
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
  const deploymentRows = buildDeploymentComponentRows(props.deploymentStatus, props.modelCatalog);
  const modelActionRows = buildModelActionRows(props.deploymentStatus, props.modelCatalog);

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
                <div className="deployment-model-cell" role="cell">
                  <strong>{modelDisplayName(row.component, props.modelCatalog)}</strong>
                  <ModelSemanticChips
                    catalog={props.modelCatalog}
                    className="deployment-model-semantics"
                    compact
                    model={row.component}
                  />
                </div>
                <span role="cell">{row.directory}</span>
                <span role="cell">{row.env}</span>
                <span role="cell">{row.files}</span>
                <span role="cell">{row.checkpoints}</span>
              </div>
            ))}
          </div>
        ) : null}
        {modelActionRows.length > 0 ? (
          <section className="model-action-grid" aria-label="模型可用性行动队列">
            {modelActionRows.map((row) => (
              <article className={`model-action-card ${row.tone}`} key={row.value}>
                <div className="model-action-card-head">
                  <strong>{row.label}</strong>
                  <span>{row.stateLabel}</span>
                </div>
                <p>{row.nextAction}</p>
                {row.constraints.length > 0 ? (
                  <div className="model-action-tags">
                    {row.constraints.map((tag) => (
                      <span key={tag}>{tag}</span>
                    ))}
                  </div>
                ) : null}
              </article>
            ))}
          </section>
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
        <PanelTitle eyebrow="Evaluation" title="辅助评估配置" />
        <div className="support-checklist compact-stack">
          <article className="support-check-item">
            <strong>状态</strong>
            <p>{props.advisorReady ? `已配置：${props.advisorState.model}` : props.advisorState.message}</p>
          </article>
          <article className="support-check-item">
            <strong>用途</strong>
            <p>生成任务评估草稿。</p>
          </article>
          <button className="ghost-button" onClick={() => void props.openAdvisorSettings()} disabled={props.advisorConfigLoading} type="button">
            {props.advisorConfigLoading ? "读取中..." : "打开配置"}
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
        <PanelTitle eyebrow="Operator" title="操作状态" />
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
  modelCatalog: ModelCatalogItem[];
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
  const totalLogLines = useMemo(() => countLogLines(props.selectedJob.logs), [props.selectedJob.logs]);
  const logKeywordHits = useMemo(
    () => countLogKeywordHits(props.selectedJob.logs, normalizedLogQuery),
    [normalizedLogQuery, props.selectedJob.logs]
  );
  const outputSections = buildOutputSections(props.selectedJob.outputs, job.model);
  const progress = props.selectedJob.phase_display;
  const advisorSuggested = isAdvisorSuggested(job.status);
  const advisorReport = props.selectedJob.advisor_report ?? null;
  const attentionJob = job.status === "failed" || job.status === "cancelled";
  const batchActionBusy = props.actionKey?.startsWith("batch:") ?? false;
  const inspectorRhythm = buildInspectorRhythm(props.selectedJob, latestLogLine, criticalLogLine, advisorReport);

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
              <span className="hero-tag">{modelDisplayName(job.model, props.modelCatalog)}</span>
              <span className="hero-tag">{sourceTypeLabel(job.source_type)}</span>
            </div>
            <ModelSemanticChips
              catalog={props.modelCatalog}
              className="detail-model-semantics"
              compact
              model={job.model}
            />
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

      <article className={`advisor-recommendation compact ${advisorSuggested ? "active" : ""}`}>
        <div>
          <span className="mini-label">辅助评估</span>
          <strong>{buildAdvisorCompactStatus(props.advisorState, advisorReport, advisorSuggested)}</strong>
        </div>
        <div className="advisor-workbench-actions">
          {!props.advisorState.enabled || !props.advisorState.configured ? (
            <button onClick={props.onConfigureAdvisor} type="button">
              配置
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
            {props.actionKey === "advisor" ? "生成中..." : "生成草稿"}
          </button>
        </div>
      </article>

      <div className="action-row">
        <button
          disabled={!props.canDispatch || props.actionKey === "dispatch" || batchActionBusy}
          onClick={() => props.onAction(`/api/jobs/${job.job_id}/dispatch`, "dispatch")}
          type="button"
        >
          运行
        </button>
        <button
          disabled={props.running || props.actionKey === "retry" || batchActionBusy}
          onClick={() => props.onAction(`/api/jobs/${job.job_id}/retry`, "retry")}
          type="button"
        >
          重试
        </button>
        <button
          disabled={props.actionKey === "duplicate" || batchActionBusy}
          onClick={() => props.onAction(`/api/jobs/${job.job_id}/duplicate`, "duplicate")}
          type="button"
        >
          复制
        </button>
        <button
          className="danger"
          disabled={!props.running || props.actionKey === "cancel" || batchActionBusy}
          onClick={() => props.onAction(`/api/jobs/${job.job_id}/cancel`, "cancel")}
          type="button"
        >
          取消
        </button>
        <a className="ghost-button" download href={props.assetUrl(`/api/jobs/${job.job_id}/bundle`)}>
          导出包
        </a>
      </div>

      {attentionJob ? (
        <article className={`attention-inspector ${job.status}`}>
          <div>
            <span className="mini-label">Attention</span>
            <strong>{job.status === "failed" ? "失败任务优先排查" : "已取消任务复查"}</strong>
            <p>{buildAttentionJobMessage(job.status, job.error_message, criticalLogLine, latestLogLine)}</p>
          </div>
          <div className="attention-inspector-actions">
            {job.error_message ? (
              <button className="ghost-button small" onClick={() => void props.onCopy(job.error_message ?? "", "错误原因")} type="button">
                复制错误
              </button>
            ) : null}
            {criticalLogLine ? (
              <button className="ghost-button small" onClick={() => void props.onCopy(criticalLogLine, "可疑日志")} type="button">
                复制可疑行
              </button>
            ) : null}
            {latestLogLine ? (
              <button className="ghost-button small" onClick={() => void props.onCopy(latestLogLine, "最新日志")} type="button">
                复制最新日志
              </button>
            ) : null}
            <button
              className="ghost-button small"
              disabled={props.running || props.actionKey === "retry" || batchActionBusy}
              onClick={() => props.onAction(`/api/jobs/${job.job_id}/retry`, "retry")}
              type="button"
            >
              {props.actionKey === "retry" ? "重试中..." : "重试任务"}
            </button>
          </div>
        </article>
      ) : null}

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
            辅助评估
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
              </div>
            </div>
            <SummaryPanel summary={props.selectedJob.result_summary} />
          </article>

          <article className="soft-panel inspector-panel" id="job-outputs-panel">
            <div className="section-head">
              <div>
                <h4>输出结果</h4>
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
                                  note: section.key === "masks" ? "动态掩膜" : undefined
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
                                  kind: "video"
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
                                      note: section.key === "masks" ? "动态掩膜" : undefined
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
              </div>
              <div className="logs-head-actions">
                <span className="section-pill">
                  {normalizedLogQuery
                    ? `命中 ${logKeywordHits}/${totalLogLines} 行 · ${filteredLogs.length}/${props.selectedJob.logs.length} 份`
                    : `${props.selectedJob.logs.length} 份`}
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
                    <pre>
                      <HighlightedLogTail query={normalizedLogQuery} text={log.tail || "暂无日志。"} />
                    </pre>
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
          <article className="soft-panel inspector-panel inspector-rhythm-panel">
            <div className="section-head">
              <div>
                <h4>检查顺序</h4>
              </div>
            </div>
            <div className="inspector-rhythm-list">
              {inspectorRhythm.map((item) => (
                <button
                  className={`inspector-rhythm-card ${item.tone}`}
                  key={item.id}
                  onClick={() => scrollToInspectorSection(item.id)}
                  type="button"
                >
                  <div>
                    <span>{item.label}</span>
                    <strong>{item.status}</strong>
                  </div>
                  <p>{item.detail}</p>
                </button>
              ))}
            </div>
          </article>

          <article className="soft-panel inspector-panel">
            <div className="section-head">
              <div>
                <h4>检查器快照</h4>
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
            <h4>辅助评估</h4>
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
                        kind: "image"
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

function buildInspectorRhythm(
  selectedJob: JobPayload,
  latestLogLine: string,
  criticalLogLine: string,
  advisorReport: AdvisorReport | null
) {
  const summary = selectedJob.result_summary;
  const evaluation = selectedJob.evaluation;
  const outputCount = selectedJob.outputs.length;
  const primaryCount = summary?.primary_artifacts?.length ?? 0;
  const logCount = selectedJob.logs.length;

  return [
    {
      id: "job-summary-panel",
      label: "摘要",
      status: summary ? "已生成" : "等待结果",
      detail: summary?.next_actions?.length ? `下一步 ${summary.next_actions.length} 条` : summary ? "有结果摘要可复查" : "结果回传后生成摘要",
      tone: summary ? "ready" : "pending"
    },
    {
      id: "job-outputs-panel",
      label: "证据",
      status: outputCount > 0 ? `${outputCount} 个产物` : "无产物",
      detail: primaryCount > 0 ? `核心检查对象 ${primaryCount} 个` : outputCount > 0 ? "按用途分组查看" : "等待远端输出下载",
      tone: outputCount > 0 ? "ready" : "pending"
    },
    {
      id: "job-logs-panel",
      label: "日志",
      status: criticalLogLine ? "待排查" : logCount > 0 ? "可追踪" : "暂无日志",
      detail: criticalLogLine || latestLogLine || "运行开始后会显示日志尾部",
      tone: criticalLogLine ? "attention" : logCount > 0 ? "ready" : "pending"
    },
    {
      id: "job-evaluation-panel",
      label: "评分",
      status: evaluation ? "人工已评" : advisorReport ? "自动已评" : "待评估",
      detail: evaluation
        ? "人工评分"
        : advisorReport
          ? `结论：${advisorReport.readiness}`
          : "等待评分",
      tone: evaluation || advisorReport ? "ready" : "pending"
    }
  ];
}

function buildAdvisorCompactStatus(state: AdvisorStatus, report: AdvisorReport | null, suggested: boolean) {
  if (!state.enabled) {
    return "未启用";
  }
  if (!state.configured) {
    return "未配置";
  }
  if (report) {
    return "已有草稿";
  }
  return suggested ? "可生成草稿" : "等待结果";
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

function HighlightedLogTail(props: { text: string; query: string }) {
  if (!props.query) {
    return <>{props.text}</>;
  }
  const pattern = new RegExp(`(${escapeRegExp(props.query)})`, "gi");
  const parts = props.text.split(pattern);
  return (
    <>
      {parts.map((part, index) =>
        part.toLowerCase() === props.query.toLowerCase() ? (
          <mark className="log-hit" key={`${part}-${index}`}>
            {part}
          </mark>
        ) : (
          <span key={`${part}-${index}`}>{part}</span>
        )
      )}
    </>
  );
}

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function countLogLines(logs: JobPayload["logs"]) {
  return logs.reduce((total, log) => {
    const lines = (log.tail || "").split(/\r?\n/).filter((line) => line.trim().length > 0);
    return total + lines.length;
  }, 0);
}

function countLogKeywordHits(logs: JobPayload["logs"], query: string) {
  if (!query) {
    return 0;
  }
  const needle = query.toLowerCase();
  return logs.reduce((total, log) => {
    const lines = (log.tail || "").split(/\r?\n/).filter((line) => line.trim().length > 0);
    return total + lines.filter((line) => line.toLowerCase().includes(needle)).length;
  }, 0);
}

function buildAttentionJobMessage(status: string, errorMessage: string | null | undefined, criticalLogLine: string, latestLogLine: string) {
  if (errorMessage) {
    return errorMessage;
  }
  if (criticalLogLine) {
    return `优先查看可疑日志：${criticalLogLine}`;
  }
  if (latestLogLine) {
    return `没有明确错误行，先从最新日志判断是否需要重试：${latestLogLine}`;
  }
  return status === "cancelled" ? "任务已取消，先确认是否需要清理远端残留或复制任务重跑。" : "任务失败但没有回传明确日志，先检查远端状态和 dispatch 日志。";
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
      description: model === "monst3r" ? "三维场景、点云、主要导出。" : "点云与主要可视化。",
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
      description: "动态 mask 与扩张 mask。",
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
      description: "未归类文件。",
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

function Fast3rParams(props: {
  formState: FormState;
  updateFormField: (key: keyof FormState, value: string) => void;
}) {
  return (
    <div className="param-grid">
      {Object.keys(defaultFast3rParams).map((key) => (
        <ParamField
          key={key}
          name={key}
          value={props.formState[key as keyof FormState]}
          choices={fast3rParamChoices[key as keyof typeof defaultFast3rParams]}
          onChange={(value) => props.updateFormField(key as keyof FormState, value)}
        />
      ))}
    </div>
  );
}

function Spann3rParams() {
  return (
    <div className="param-static-note">
      <strong>当前 Runner 使用固定序列参数</strong>
      <p>提交时使用后端默认序列配置。</p>
      <div className="param-static-grid">
        <span>resolution 224</span>
        <span>kf_every 10</span>
        <span>conf_thresh 0.001</span>
        <span>offline false</span>
      </div>
    </div>
  );
}

function CatalogOnlyParams(props: { blocker: string | null }) {
  return (
    <div className="param-static-note blocked">
      <strong>当前模型暂不进入创建队列</strong>
      <p>{props.blocker ?? "该目录模型还没有可派发 runner、部署合同或 smoke 结果。"}</p>
    </div>
  );
}

function ParamField(props: {
  name: string;
  value: string;
  choices?: ParamChoice[];
  onChange: (value: string) => void;
}) {
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

function AdvisorPanel(props: { report: AdvisorReport | null }) {
  if (!props.report) {
    return <span className="muted-text">暂无自动评估。</span>;
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
          <strong>辅助评估未就绪</strong>
          <p>{props.advisorState.message}</p>
          <div className="advisor-workbench-actions">
            <button onClick={props.onConfigure} type="button">
              配置
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!props.job) {
    return (
      <div className={`advisor-workbench ${props.compact ? "compact" : ""}`}>
        <div className="empty-state">先选中一条任务。</div>
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
            {props.actionKey === "advisor" ? "生成中..." : report ? "重新生成" : "生成草稿"}
          </button>
        </div>
      </div>

      {!report ? (
        <div className="advisor-config-note">
          <strong>{suggested ? "可生成评估草稿" : "等待任务结束"}</strong>
          <p>{suggested ? "会读取参数、摘要和日志。" : "完成、失败或取消后再生成。"}</p>
        </div>
      ) : (
        <>
          <AdvisorPanel report={report} />
          {props.compact ? null : (
            <div className="advisor-copy-row">
              <button onClick={() => void props.onCopy(report.summary, "评估摘要")} type="button">
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

export default App;
