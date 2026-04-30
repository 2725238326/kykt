import { FormEvent, KeyboardEvent as ReactKeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type {
  AdvisorConfig,
  AdvisorDiagnostics,
  AdvisorProvider,
  AdvisorStatus,
  AppState,
  BackendStatusPayload,
  BootstrapPayload,
  DeploymentStatusPayload,
  DevelopmentLaneItem,
  EvaluationPayload,
  JobPayload,
  JobsListPayload,
  ModelContract,
  ResultContract,
  SamplesPayload,
  ValidationCreateResponse
} from "./types";
import { API_BASE, DEFAULT_BOOTSTRAP } from "./appConfig";
import {
  backendStatusText,
  delay,
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
  batchActionLabel,
  buildActionMessage,
  buildAdvisorChecklist,
  buildCaptureChecklist,
  buildSystemChecklist,
  canDispatchJobStatus,
  isAdvisorSuggested,
  matchesJobQuery
} from "./workflowHelpers";
import type { BatchJobAction, JobListItem } from "./workflowHelpers";
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
import { SampleMatrixPanel } from "./SampleMatrixPanel";
import { AdvisorWorkbench } from "./AdvisorWorkbench";
import { JobDetail } from "./JobDetail";
import type { PreviewAsset } from "./JobDetail";
import { currentStepLabel } from "./jobInspectorHelpers";
import { DevelopmentCyclePanel } from "./DevelopmentCyclePanel";
import { ResearchAccelerationPanel } from "./ResearchAccelerationPanel";
import { DynamicParamForm } from "./DynamicParamForm";

type ServiceState = "starting" | "ready" | "degraded";
type JobFilter = "all" | "running" | "attention" | "finished";
type WorkspaceTab = "overview" | "create" | "jobs" | "advisor" | "system";

const workspaceTabs: Array<{ key: WorkspaceTab; label: string; note: string }> = [
  { key: "overview", label: "工作台", note: "全局状态与焦点任务" },
  { key: "create", label: "文件与新建", note: "整理输入并创建任务" },
  { key: "jobs", label: "运行与结果", note: "筛选任务、跟进状态、查看产物" },
  { key: "advisor", label: "辅助评估", note: "自动评估草稿" },
  { key: "system", label: "帮助与系统", note: "服务与远端部署" }
];

function App() {
  const [appState, setAppState] = useState<AppState | null>(null);
  const [jobs, setJobs] = useState<JobsListPayload["jobs"]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<JobPayload | null>(null);
  const [samplesPayload, setSamplesPayload] = useState<SamplesPayload | null>(null);
  const [samplesError, setSamplesError] = useState<string | null>(null);
  const [developmentLanes, setDevelopmentLanes] = useState<DevelopmentLaneItem[]>([]);
  const [developmentLaneError, setDevelopmentLaneError] = useState<string | null>(null);
  const [creatingDevelopmentLane, setCreatingDevelopmentLane] = useState(false);
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
  const [advisorProviders, setAdvisorProviders] = useState<AdvisorProvider[]>([]);
  const [advisorDiagnostics, setAdvisorDiagnostics] = useState<AdvisorDiagnostics | null>(null);
  const [advisorForm, setAdvisorForm] = useState<AdvisorConfig>({
    enabled: false,
    baseUrl: "",
    apiKey: "",
    model: "gpt-4o-mini",
    maxTokens: 2048,
    systemPrompt: "",
    structuredOutput: true,
    timeoutSeconds: 60,
  });
  const [validationResponse, setValidationResponse] = useState<ValidationCreateResponse | null>(null);
  const recoveryInFlightRef = useRef(false);
  const jobSearchInputRef = useRef<HTMLInputElement | null>(null);
  const [formState, setFormState] = useState<{
    model: string;
    source_type: string;
    notes: string;
    params: Record<string, any>;
  }>({
    model: "dust3r",
    source_type: "images",
    notes: "",
    params: {}
  });

  const modelCatalog = useMemo(() => appState?.modelCatalog ?? [], [appState]);
  const modelContracts = useMemo(() => appState?.modelContracts ?? {}, [appState]);
  const advisorState = useMemo(() => appState?.advisor ?? {
    enabled: false,
    configured: false,
    base_url: "",
    model: "",
    has_api_key: false,
    message: "辅助评估尚未配置。"
  }, [appState]);

  const serviceReady = serviceState === "ready";
  const advisorReady = advisorState.enabled && advisorState.configured;

  function openPreviewAsset(asset: PreviewAsset) {
    setPreviewAsset(asset);
  }

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

  const selectedModelContract = useMemo(
    () => modelContracts[formState.model] ?? null,
    [formState.model, modelContracts]
  );

  const selectedModelSourceTypes = useMemo(
    () => selectedModelContract?.allowedSourceTypes ?? selectedModelCatalog?.source_types ?? [],
    [selectedModelCatalog, selectedModelContract]
  );

  const selectedModelLaunchBlocker = useMemo(
    () => selectedModelContract?.launchBlocker ?? selectedModelCatalog?.launch_blocker ?? null,
    [selectedModelCatalog, selectedModelContract]
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

  const summary = useMemo(() => appState?.summary ?? {
    total: jobs.length,
    running: jobs.filter((item) => item.job.status === "running").length,
    finished: jobs.filter((item) => item.job.status === "finished").length,
    failed: jobs.filter((item) => item.job.status === "failed").length,
    cancelled: jobs.filter((item) => item.job.status === "cancelled").length
  }, [appState, jobs]);

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

  async function refreshAppState() {
    const state = await fetchJson<AppState>("/api/app/state");
    setAppState(state);
    if (state.developmentLanes) {
      setDevelopmentLanes(state.developmentLanes);
    }
    return state;
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

      await refreshAppState();
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

  useEffect(() => {
    let cancelled = false;

    async function boot() {
      await loadDesktopBackendStatus();

      for (let attempt = 1; attempt <= 35; attempt += 1) {
        if (cancelled) {
          return;
        }

        try {
          await refreshAppState();
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

  useEffect(() => {
    if (activeWorkspace !== "create" || !formState.model || !serviceReady) {
      return;
    }
    void validateLaunch();
  }, [activeWorkspace, formState.model, formState.source_type, files.length, serviceReady]);

  async function validateLaunch() {
    try {
      const res = await fetchJson<ValidationCreateResponse>(`/api/models/${formState.model}/validate-create`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sourceType: formState.source_type,
          fileCount: files.length
        })
      });
      setValidationResponse(res);
    } catch {
      setValidationResponse(null);
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
      const [config, providers, diagnostics] = await Promise.all([
        fetchJson<AdvisorConfig>("/api/advisor/config"),
        fetchJson<AdvisorProvider[]>("/api/advisor/providers"),
        fetchJson<AdvisorDiagnostics>("/api/advisor/diagnostics")
      ]);
      setAdvisorForm(config);
      setAdvisorProviders(providers);
      setAdvisorDiagnostics(diagnostics);
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
      const payload = await fetchJson<AdvisorStatus>("/api/advisor/config", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(advisorForm),
      });
      await refreshAppState();
      setAdvisorModalOpen(false);
      setInfoMessage(payload.configured ? "辅助评估配置已保存并就绪。" : "评估配置已保存，但尚未通过诊断。");
    } catch (error) {
      setErrorMessage(friendlyError(error, "保存 AI 配置失败。"));
    } finally {
      setAdvisorConfigSaving(false);
    }
  }

  async function testAdvisorConnection() {
    setAdvisorConfigSaving(true);
    try {
      await fetchJson("/api/advisor/test", { method: "POST" });
      setInfoMessage("AI 助手连接测试成功。");
    } catch (error) {
      setErrorMessage(friendlyError(error, "连接测试失败。"));
    } finally {
      setAdvisorConfigSaving(false);
    }
  }

  async function loadJobDetail(jobId: string, showError = true) {
    try {
      const detail = await fetchJson<JobPayload>(`/api/jobs/${jobId}`);
      if (!detail.contract) {
        const contract = await fetchJson<ResultContract>(`/api/jobs/${jobId}/contract`);
        detail.contract = contract;
      }
      setSelectedJob(detail);
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

  function updateFormField(key: string, value: any) {
    if (key === "model" || key === "source_type" || key === "notes") {
      setFormState((current) => ({ ...current, [key]: value }));
    } else {
      setFormState((current) => ({
        ...current,
        params: { ...current.params, [key]: value }
      }));
    }
  }

  async function updateModel(value: string) {
    setFormState((current) => ({
      ...current,
      model: value,
      params: {}
    }));

    if (!modelContracts[value]) {
      try {
        const contract = await fetchJson<ModelContract>(`/api/models/${value}/contract`);
        setAppState((current) => {
          if (!current) return null;
          return {
            ...current,
            modelContracts: { ...current.modelContracts, [value]: contract }
          };
        });
        
        const nextSourceType = contract.allowedSourceTypes.includes(formState.source_type) 
          ? formState.source_type 
          : contract.allowedSourceTypes[0] ?? "images";
        
        setFormState((current) => ({
          ...current,
          source_type: nextSourceType,
          params: contract.paramSchema.fields.reduce((acc, field) => {
            acc[field.key] = field.default;
            return acc;
          }, {} as Record<string, any>)
        }));
      } catch (error) {
        setErrorMessage(`加载模型 ${value} 的配置合同失败。`);
      }
    }
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

  function applyPreset(presetKey: string) {
    const contract = modelContracts[formState.model];
    if (!contract || !contract.paramSchema.presets) return;
    
    const presetValues = contract.paramSchema.presets[presetKey];
    if (presetValues) {
      setFormState((current) => ({
        ...current,
        params: { ...current.params, ...presetValues }
      }));
    }
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

    if (validationResponse && !validationResponse.ok) {
      setErrorMessage(validationResponse.errors.join("；"));
      return;
    }

    setSubmitting(true);
    const formData = new FormData();
    formData.append("model", formState.model);
    formData.append("source_type", formState.source_type);
    formData.append("notes", formState.notes);
    formData.append("params", JSON.stringify(formState.params));
    
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

  async function createDevelopmentLaneSeed(category: DevelopmentLaneItem["category"]) {
    const presets: Record<DevelopmentLaneItem["category"], Partial<DevelopmentLaneItem>> = {
      paper_reproduction: {
        title: "新论文复现计划",
        category: "paper_reproduction",
        status: "draft",
        priority: "P2",
        nextAction: "补充论文链接、官方仓库、权重位置和首个 smoke 样例。",
        blockers: [],
        mergeTarget: "deferred_research"
      },
      model_runner: {
        title: "新模型 Runner 接入",
        category: "model_runner",
        status: "draft",
        priority: "P1",
        nextAction: "记录 runnerPath、environmentPath 和标准输入输出契约。",
        blockers: [],
        mergeTarget: "runner"
      },
      prototype: {
        title: "新原型验证",
        category: "prototype",
        status: "draft",
        priority: "P2",
        nextAction: "定义最小可运行样例和 smoke test 产物。",
        blockers: [],
        mergeTarget: "runner"
      },
      evaluation: {
        title: "新评测流设计",
        category: "evaluation",
        status: "draft",
        priority: "P2",
        nextAction: "对齐评分维度、样例矩阵和 Advisor 输出格式。",
        blockers: [],
        mergeTarget: "advisor"
      },
      ui_workflow: {
        title: "新 UI 工作流原型",
        category: "ui_workflow",
        status: "draft",
        priority: "P3",
        nextAction: "明确入口、状态流转和需要的后端 API。",
        blockers: [],
        mergeTarget: "report"
      }
    };

    setCreatingDevelopmentLane(true);
    try {
      const created = await fetchJson<DevelopmentLaneItem>("/api/development/lanes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(presets[category])
      });
      setDevelopmentLanes((current) => [created, ...current]);
      setInfoMessage(`研发车道已创建：${created.title}`);
    } catch (error) {
      setErrorMessage(friendlyError(error, "创建研发车道失败。"));
    } finally {
      setCreatingDevelopmentLane(false);
    }
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

      if (lastPayload) {
        setSelectedJob(lastPayload);
        setSelectedJobId(lastPayload.job.job_id);
      }
      await loadJobs(false);

      if (failures.length > 0) {
        setErrorMessage(`批量${label}完成 ${completed}/${targetJobs.length} 个，失败 ${failures.length} 个。`);
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

  function assetUrl(path: string) {
    if (/^(https?:|data:|blob:)/.test(path)) {
      return path;
    }
    return `${API_BASE}${path.startsWith("/") ? "" : "/"}${path}`;
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
            developmentLanes={developmentLanes}
            developmentLaneError={developmentLaneError}
            modelCatalog={modelCatalog}
            openWorkspace={openWorkspace}
            activeJob={activeJob}
            advisorState={advisorState}
            actionKey={actionKey}
            postJobAction={postJobAction}
            openAdvisorSettings={openAdvisorSettings}
            copyText={copyText}
            onCreateDevelopmentLane={createDevelopmentLaneSeed}
            creatingDevelopmentLane={creatingDevelopmentLane}
          />
        ) : null}

        {activeWorkspace === "create" ? (
          <section className="layout-grid create-layout">
            <article className="panel create-panel">
              <PanelTitle eyebrow="新建任务" title="选择模型和输入" />
              <div className="create-run-strip">
                <article className={`create-run-item ${serviceReady ? "ready" : "starting"}`}>
                  <span>服务状态</span>
                  <strong>{serviceReady ? "已就绪" : "准备中"}</strong>
                </article>
                <article className={`create-run-item ${selectedModelLaunchBlocker ? "danger" : "ready"}`}>
                  <span>模型可用性</span>
                  <strong>{selectedModelLaunchBlocker ? "不可用" : "就绪"}</strong>
                </article>
                <article className={`create-run-item ${validationResponse?.ok ? "ready" : "neutral"}`}>
                  <span>输入校验</span>
                  <strong>{validationResponse?.ok ? "通过" : "等待输入"}</strong>
                </article>
              </div>

              <form className="form-stack" onSubmit={handleCreateJob}>
                <section className="create-workbench-grid">
                  <article className="create-block create-config-block">
                    <div className="create-block-head">
                      <div>
                        <span className="mini-label">任务配置</span>
                        <strong>模型与输入来源</strong>
                      </div>
                    </div>

                    <div className="form-row">
                      <label className="field">
                        <span>模型</span>
                        <select value={formState.model} onChange={(e) => updateModel(e.target.value)}>
                          {runnableModelCatalog.map((m) => (
                            <option key={m.value} value={m.value}>{m.label}</option>
                          ))}
                          {catalogOnlyModelCatalog.map((m) => (
                            <option key={m.value} value={m.value} disabled>{m.label} (未部署)</option>
                          ))}
                        </select>
                      </label>
                      <label className="field">
                        <span>输入类型</span>
                        <select
                          value={formState.source_type}
                          onChange={(e) => updateFormField("source_type", e.target.value)}
                        >
                          {appState?.sourceTypes.filter(st => selectedModelSourceTypes.includes(st.value)).map((st) => (
                            <option key={st.value} value={st.value}>{st.label}</option>
                          ))}
                        </select>
                      </label>
                    </div>

                    <div className="create-model-context">
                      <strong>{selectedModelCatalog?.description}</strong>
                      <ModelSemanticChips catalog={modelCatalog} model={formState.model} className="create-model-facts" compact />
                      {selectedModelLaunchBlocker && <p className="create-model-note blocked">{selectedModelLaunchBlocker}</p>}
                    </div>

                    <label className="field">
                      <span>备注</span>
                      <textarea
                        rows={3}
                        value={formState.notes}
                        onChange={(e) => updateFormField("notes", e.target.value)}
                        placeholder="任务备注..."
                      />
                    </label>
                  </article>

                  <article className="create-block create-staging-block">
                    <div className="create-block-head">
                      <strong>本地待上传文件</strong>
                      <span className="section-pill">{files.length} 个文件</span>
                    </div>
                    <label className="dropzone">
                      <input type="file" multiple onChange={(e) => setFiles(Array.from(e.target.files ?? []))} />
                      <span>点击或拖拽选择文件</span>
                    </label>
                    <div className="staging-stats">
                      <SummaryStat label="总大小" value={formatFileSize(pendingTotalSize)} />
                      <SummaryStat label="类型分布" value={pendingTypeSummary} />
                    </div>
                    {files.length > 0 && (
                      <div className="staging-table">
                        {files.map((file) => (
                          <div className="staging-row" key={`${file.name}-${file.size}`}>
                            <span className="staging-name">{file.name}</span>
                            <span>{pendingFileRoleLabel(file)}</span>
                            <button className="ghost-button small" type="button" onClick={() => removePendingFile(file.name, file.size)}>移除</button>
                          </div>
                        ))}
                      </div>
                    )}
                  </article>

                  <article className="create-block create-params-block">
                    <details className="advanced-panel" open>
                      <summary>高级参数</summary>
                      {selectedModelContract?.paramSchema.presets && (
                        <div className="preset-strip">
                          <div className="preset-pills">
                            {Object.entries(selectedModelContract.paramSchema.presets).map(([key, p]) => (
                              <button key={key} className="preset-pill" type="button" onClick={() => applyPreset(key)}>
                                <strong>{key}</strong>
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                      {selectedModelContract ? (
                        <DynamicParamForm
                          fields={selectedModelContract.paramSchema.fields}
                          values={formState.params}
                          onChange={(k, v) => updateFormField(k, v)}
                        />
                      ) : (
                        <div className="empty-state">加载参数契约中...</div>
                      )}
                    </details>
                  </article>
                </section>

                <div className="create-submit-dock">
                  <div className="create-validation-info">
                    {validationResponse && !validationResponse.ok && (
                      <div className="validation-errors">
                        {validationResponse.errors.map(err => <p key={err} className="text-danger">{err}</p>)}
                      </div>
                    )}
                  </div>
                  <button className="primary-button" disabled={!serviceReady || submitting || !!selectedModelLaunchBlocker || !validationResponse?.ok} type="submit">
                    {submitting ? "创建中..." : "启动任务"}
                  </button>
                </div>
              </form>
            </article>

            <aside className="panel create-support-panel">
              <PanelTitle eyebrow="输入规范" title="检查清单" />
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
                      ? "暂无任务。"
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
                <div className="empty-state large">选择一个任务查看详情。</div>
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
            appState={appState}
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
                <strong>AI Advisor 设置</strong>
              </div>
              <button className="ghost-button small" onClick={() => setAdvisorModalOpen(false)} type="button">
                关闭
              </button>
            </div>

            <form className="form-stack settings-form" onSubmit={saveAdvisorSettings}>
              <label className="field">
                <span>启用辅助评估</span>
                <input type="checkbox" checked={advisorForm.enabled} onChange={e => setAdvisorForm(c => ({...c, enabled: e.target.checked}))} />
              </label>

              <label className="field">
                <span>提供商 (Provider)</span>
                <select value={advisorForm.model} onChange={e => setAdvisorForm(c => ({...c, model: e.target.value}))}>
                  {advisorProviders.map(p => (
                    <optgroup key={p.id} label={p.label}>
                      {p.models.map(m => <option key={m} value={m}>{m}</option>)}
                    </optgroup>
                  ))}
                </select>
              </label>

              <label className="field">
                <span>Base URL</span>
                <input value={advisorForm.baseUrl} onChange={e => setAdvisorForm(c => ({...c, baseUrl: e.target.value}))} />
              </label>

              <label className="field">
                <span>API Key</span>
                <input type="password" value={advisorForm.apiKey} onChange={e => setAdvisorForm(c => ({...c, apiKey: e.target.value}))} placeholder={advisorForm.hasApiKey ? "已保存" : "输入 Key"} />
              </label>

              <div className="diagnostics-summary">
                {advisorDiagnostics?.checks.map(c => (
                  <div key={c.name} className={`diag-item ${c.passed ? "ok" : "fail"}`}>
                    <span>{c.name}</span>
                    <small>{c.message}</small>
                  </div>
                ))}
              </div>

              <div className="settings-modal-actions">
                <button className="ghost-button" type="button" onClick={testAdvisorConnection}>测试连接</button>
                <button className="primary-button" disabled={advisorConfigSaving} type="submit">保存配置</button>
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
              </div>
              <button className="ghost-button small" onClick={() => setPreviewAsset(null)} type="button">关闭</button>
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
  developmentLanes: DevelopmentLaneItem[];
  developmentLaneError: string | null;
  modelCatalog: ModelCatalogItem[];
  openWorkspace: (workspace: WorkspaceTab, jobId?: string) => void;
  activeJob: JobPayload | null;
  advisorState: AdvisorStatus;
  actionKey: string | null;
  postJobAction: (path: string, key: string) => Promise<void>;
  openAdvisorSettings: () => Promise<void> | void;
  copyText: (value: string, label: string) => Promise<void>;
  onCreateDevelopmentLane: (category: DevelopmentLaneItem["category"]) => Promise<void>;
  creatingDevelopmentLane: boolean;
}) {
  const focusJob = props.focusJob;
  const runnableCount = props.activeModelCatalog.filter((item) => item.runnable).length;
  const blockedModels = props.activeModelCatalog.filter((item) => !item.runnable);

  return (
    <>
      <div className="overview-focus-strip">
        <article className="focus-strip-item">
          <span className="mini-label">活跃任务</span>
          <strong>{props.runningJobs.length} Running</strong>
        </article>
        <article className="focus-strip-item">
          <span className="mini-label">需要关注</span>
          <strong className={props.attentionJobs.length > 0 ? "text-danger" : ""}>
            {props.attentionJobs.length} Attention
          </strong>
        </article>
        <article className="focus-strip-item">
          <span className="mini-label">模型就绪</span>
          <strong>{runnableCount} / {props.activeModelCatalog.length}</strong>
        </article>
        <article className="focus-strip-item">
          <span className="mini-label">评估建议</span>
          <strong>{props.advisorState.configured ? "Ready" : "Not Configured"}</strong>
        </article>
        <article className="focus-strip-item next-action-item">
          <span className="mini-label">Next Action</span>
          <strong>验证 MASt3R 新版损失函数</strong>
        </article>
      </div>

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
          <PanelTitle eyebrow="Runtime" title="调度状态" />
          <div className="kpi-grid">
            <MiniStat label="总任务" value={props.summary.total} />
            <MiniStat label="运行中" value={props.summary.running} />
            <MiniStat label="待处理" value={props.attentionJobs.length} />
            <MiniStat label="已完成" value={props.summary.finished} />
          </div>
          {blockedModels.length > 0 && (
            <div className="overview-callout danger">
              <span className="mini-label">阻塞模型</span>
              <strong>{blockedModels.length} 个模型由于环境缺失被阻塞</strong>
              <p>{blockedModels.map(m => m.label).join(", ")}</p>
            </div>
          )}
          <div className="overview-ops-dock">
            <button onClick={() => props.openWorkspace("jobs")} type="button">
              <span>Jobs</span>
              <strong>{props.runningJobs.length}</strong>
              <small>运行中</small>
            </button>
            <button onClick={() => props.openWorkspace("create")} type="button">
              <span>New</span>
              <small>发起任务</small>
            </button>
          </div>
        </aside>
      </section>

      <section className="overview-development-grid">
        <DevelopmentCyclePanel items={props.developmentLanes} />
        <ResearchAccelerationPanel
          items={props.developmentLanes}
          creating={props.creatingDevelopmentLane}
          onCreateSeed={props.onCreateDevelopmentLane}
        />
      </section>

      <section className="overview-support-grid workbench-overview-support-grid">
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
  appState: AppState | null;
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
          {props.appState?.summary && (
            <>
              <article className="support-check-item">
                <strong>SSH 目标</strong>
                <p>后端连接信息</p>
              </article>
            </>
          )}
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

export default App;
