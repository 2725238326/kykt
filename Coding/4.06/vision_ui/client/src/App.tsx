import { FormEvent, KeyboardEvent as ReactKeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type {
  AdvisorConfig,
  AdvisorDiagnostics,
  AdvisorProvider,
  AdvisorStatus,
  AppState,
  BackendStatusPayload,
  BatchCompareResponse,
  ComparePacket,
  DeploymentStatusPayload,
  DevelopmentLaneItem,
  EvaluationPayload,
  InspectionPacket,
  JobPayload,
  JobsListPayload,
  ModelContract,
  ResultContract,
  SamplesPayload,
  ValidationCreateResponse
} from "./types";
import { API_BASE } from "./appConfig";
import {
  backendStatusText,
  delay,
  formatDateTime,
  formatParamLabel,
  friendlyError,
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
  MessageBanner,
  ModelSemanticChips,
  PanelTitle,
  StatusBadge,
  SummaryStat,
  MiniStat
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
import { Sidebar } from "./Sidebar";
import { CommandBar } from "./CommandBar";
import { QueueWorkspace } from "./QueueWorkspace";
import { InspectWorkspace } from "./InspectWorkspace";
import { CompareBoard, CompareBoardInline } from "./CompareBoard";
import { filterRunnableModels, modelCompatibilityHint } from "./compareHelpers";

export type ServiceState = "starting" | "ready" | "degraded";
export type JobFilter = "all" | "running" | "attention" | "finished";
export type WorkspaceTab = "queue" | "create" | "inspect" | "samples" | "compare" | "development" | "system";
export type CreateMode = "single" | "batch";

function App() {
  const [appState, setAppState] = useState<AppState | null>(null);
  const [jobs, setJobs] = useState<JobsListPayload["jobs"]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [selectedInspection, setSelectedInspection] = useState<InspectionPacket | null>(null);
  const [samplesPayload, setSamplesPayload] = useState<SamplesPayload | null>(null);
  const [samplesError, setSamplesError] = useState<string | null>(null);
  const [developmentLanes, setDevelopmentLanes] = useState<DevelopmentLaneItem[]>([]);
  const [deploymentStatus, setDeploymentStatus] = useState<DeploymentStatusPayload | null>(null);
  const [deploymentError, setDeploymentError] = useState<string | null>(null);
  const [deploymentLoading, setDeploymentLoading] = useState(false);
  const [activeWorkspace, setActiveWorkspace] = useState<WorkspaceTab>("queue");
  const [backendStatus, setBackendStatus] = useState<BackendStatusPayload | null>(null);
  const [serviceState, setServiceState] = useState<ServiceState>("starting");
  const [serviceMessage, setServiceMessage] = useState("正在准备本地服务...");
  const [submitting, setSubmitting] = useState(false);
  const [actionKey, setActionKey] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [infoMessage, setInfoMessage] = useState<string | null>(null);
  const [files, setFiles] = useState<File[]>([]);
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
  const [createMode, setCreateMode] = useState<CreateMode>("single");
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [batchAutoDispatch, setBatchAutoDispatch] = useState(false);
  const [batchSubmitting, setBatchSubmitting] = useState(false);
  const [compareSampleId, setCompareSampleId] = useState<string | null>(null);
  const [comparePacket, setComparePacket] = useState<ComparePacket | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareError, setCompareError] = useState<string | null>(null);

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

  const runnableModelCatalog = useMemo(() => modelCatalog.filter((item) => item.runnable), [modelCatalog]);
  const catalogOnlyModelCatalog = useMemo(() => modelCatalog.filter((item) => !item.runnable), [modelCatalog]);

  const selectedModelCatalog = useMemo(() => modelCatalog.find((item) => item.value === formState.model) ?? null, [formState.model, modelCatalog]);
  const selectedModelContract = useMemo(() => modelContracts[formState.model] ?? null, [formState.model, modelContracts]);
  const selectedModelSourceTypes = useMemo(() => selectedModelContract?.allowedSourceTypes ?? selectedModelCatalog?.source_types ?? [], [selectedModelCatalog, selectedModelContract]);
  const selectedModelLaunchBlocker = useMemo(() => selectedModelContract?.launchBlocker ?? selectedModelCatalog?.launch_blocker ?? null, [selectedModelCatalog, selectedModelContract]);

  const summary = useMemo(() => ({
    total: jobs.length,
    running: jobs.filter((item) => item.job.status === "running").length,
    finished: jobs.filter((item) => item.job.status === "finished").length,
    failed: jobs.filter((item) => item.job.status === "failed").length,
    cancelled: jobs.filter((item) => item.job.status === "cancelled").length
  }), [jobs]);

  const pendingImageCount = useMemo(() => files.filter((file) => isImageLikeFile(file)).length, [files]);
  const pendingVideoCount = useMemo(() => files.filter((file) => isVideoLikeFile(file)).length, [files]);
  const pendingTotalSize = useMemo(() => files.reduce((total, file) => total + file.size, 0), [files]);
  const pendingTypeSummary = useMemo(() => [
    pendingImageCount > 0 ? `图片 ${pendingImageCount}` : null,
    pendingVideoCount > 0 ? `视频 ${pendingVideoCount}` : null,
  ].filter(Boolean).join(" / ") || "暂无", [pendingImageCount, pendingVideoCount]);

  const batchCompatibleModels = useMemo(() => 
    filterRunnableModels(modelCatalog, formState.source_type),
    [modelCatalog, formState.source_type]
  );

  const batchModelHints = useMemo(() => {
    const hints: Record<string, string | null> = {};
    for (const m of selectedModels) {
      hints[m] = modelCompatibilityHint(m, formState.source_type, modelCatalog);
    }
    return hints;
  }, [selectedModels, formState.source_type, modelCatalog]);

  const batchHasBlockers = useMemo(() => 
    selectedModels.some((m) => batchModelHints[m] !== null),
    [selectedModels, batchModelHints]
  );

  async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE}${path}`, init);
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || `请求失败：${response.status}`);
    }
    return response.json();
  }

  async function refreshAppState() {
    try {
      const state = await fetchJson<AppState>("/api/app/state");
      setAppState(state);
      if (state.developmentLanes) setDevelopmentLanes(state.developmentLanes);
      return state;
    } catch (e) {
      console.error("Failed to refresh app state", e);
    }
  }

  async function loadJobs(showError = true) {
    try {
      const payload = await fetchJson<JobsListPayload>("/api/jobs");
      setJobs(payload.jobs);
      setServiceState("ready");
      setServiceMessage("本地服务已就绪");
    } catch (error) {
      if (showError) setErrorMessage(friendlyError(error, "加载任务列表失败。"));
      setServiceState("degraded");
      setServiceMessage("本地服务连接异常");
    }
  }

  async function loadSamples(showError = true) {
    try {
      const payload = await fetchJson<SamplesPayload>("/api/samples");
      setSamplesPayload(payload);
    } catch (error) {
      if (showError) console.error("Samples API error", error);
    }
  }

  async function loadDeploymentStatus(showError = true, refresh = false) {
    setDeploymentLoading(true);
    try {
      const payload = await fetchJson<DeploymentStatusPayload>(`/api/deployment/status${refresh ? "?refresh=true" : ""}`);
      setDeploymentStatus(payload);
    } catch (error) {
      if (showError) setErrorMessage(friendlyError(error, "远端部署状态读取失败。"));
    } finally {
      setDeploymentLoading(false);
    }
  }

  async function loadDesktopBackendStatus() {
    try {
      const status = await invoke<BackendStatusPayload>("backend_status");
      setBackendStatus(status);
    } catch {
      setBackendStatus(null);
    }
  }

  useEffect(() => {
    loadDesktopBackendStatus();
    refreshAppState();
    loadJobs();
    loadSamples();
    
    const interval = setInterval(() => {
      loadJobs(false);
      loadDesktopBackendStatus();
    }, 8000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (selectedJobId && activeWorkspace === "inspect") loadInspection(selectedJobId);
  }, [selectedJobId, activeWorkspace]);

  useEffect(() => {
    if (activeWorkspace === "create" && formState.model) validateLaunch();
  }, [activeWorkspace, formState.model, files.length]);

  async function validateLaunch() {
    try {
      const res = await fetchJson<ValidationCreateResponse>(`/api/models/${formState.model}/validate-create`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sourceType: formState.source_type, fileCount: files.length })
      });
      setValidationResponse(res);
    } catch {
      setValidationResponse(null);
    }
  }

  function handleWorkspaceChange(tab: WorkspaceTab) {
    setActiveWorkspace(tab);
    setErrorMessage(null);
    setInfoMessage(null);
  }

  async function handleInspectJob(jobId: string) {
    setSelectedJobId(jobId);
    setActiveWorkspace("inspect");
    await loadInspection(jobId);
  }

  async function loadInspection(jobId: string, showError = true) {
    try {
      const packet = await fetchJson<InspectionPacket>(`/api/jobs/${jobId}/inspection`);
      setSelectedInspection(packet);
    } catch (error) {
      if (showError) setErrorMessage(friendlyError(error, "加载任务检视数据失败。"));
    }
  }

  async function loadComparePacket(sampleId: string, showError = true) {
    setCompareLoading(true);
    setCompareError(null);
    try {
      const packet = await fetchJson<ComparePacket>(`/api/compare/samples/${encodeURIComponent(sampleId)}`);
      setComparePacket(packet);
    } catch (error) {
      if (showError) setCompareError(friendlyError(error, "加载对比数据失败。"));
    } finally {
      setCompareLoading(false);
    }
  }

  async function handleBatchCompare(autoDispatch: boolean) {
    if (selectedModels.length < 1) {
      setErrorMessage("请至少选择一个模型。");
      return;
    }
    if (files.length === 0) {
      setErrorMessage("请先上传输入文件。");
      return;
    }
    if (batchHasBlockers) {
      setErrorMessage("部分模型不兼容当前输入类型。");
      return;
    }

    setBatchSubmitting(true);
    const formData = new FormData();
    formData.append("models", JSON.stringify(selectedModels));
    formData.append("source_type", formState.source_type);
    formData.append("notes", formState.notes);
    formData.append("auto_dispatch", autoDispatch ? "true" : "false");
    formData.append("params", JSON.stringify(formState.params));
    files.forEach(f => formData.append("files", f, f.name));

    try {
      const result = await fetchJson<BatchCompareResponse>("/api/compare/batches", {
        method: "POST",
        body: formData
      });
      setFiles([]);
      setSelectedModels([]);
      setCompareSampleId(result.sampleId);
      setComparePacket(result.compare);
      setActiveWorkspace("compare");
      setInfoMessage(`批量对比已创建：${result.models.length} 个任务，sample_id: ${result.sampleId}`);
      loadJobs(false);
    } catch (e) {
      setErrorMessage(friendlyError(e, "批量对比创建失败"));
    } finally {
      setBatchSubmitting(false);
    }
  }

  function toggleBatchModel(model: string) {
    setSelectedModels((prev) =>
      prev.includes(model) ? prev.filter((m) => m !== model) : [...prev, model]
    );
  }

  function openCompareBoard(sampleId: string) {
    setCompareSampleId(sampleId);
    loadComparePacket(sampleId);
    setActiveWorkspace("compare");
  }

  const workspaceTitle = useMemo(() => {
    switch (activeWorkspace) {
      case "queue": return "任务队列";
      case "create": return createMode === "batch" ? "批量对比" : "新建任务";
      case "inspect": return `检视：${selectedJobId || "尚未选择"}`;
      case "samples": return "样例矩阵";
      case "compare": return `对比面板：${compareSampleId || "--"}`;
      case "development": return "研发加速";
      case "system": return "系统配置";
    }
  }, [activeWorkspace, selectedJobId, createMode, compareSampleId]);

  async function openAdvisorSettings() {
    setAdvisorConfigLoading(true);
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
    } catch (e) {
      setErrorMessage("无法读取 Advisor 配置");
    } finally {
      setAdvisorConfigLoading(false);
    }
  }

  async function saveAdvisorSettings(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setAdvisorConfigSaving(true);
    try {
      await fetchJson("/api/advisor/config", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(advisorForm),
      });
      await refreshAppState();
      setAdvisorModalOpen(false);
      setInfoMessage("Advisor 配置已更新");
    } catch (e) {
      setErrorMessage("保存配置失败");
    } finally {
      setAdvisorConfigSaving(false);
    }
  }

  async function handleCommandAction(action: string) {
    if (action === "submit") handleCreateJob();
    if (action === "clear") {
      setFiles([]);
      setFormState(prev => ({ ...prev, notes: "", params: {} }));
    }
    if (action === "dispatch" && selectedJobId) postJobAction(`/api/jobs/${selectedJobId}/dispatch`, "dispatch");
    if (action === "retry" && selectedJobId) postJobAction(`/api/jobs/${selectedJobId}/retry`, "retry");
    if (action === "cancel" && selectedJobId) postJobAction(`/api/jobs/${selectedJobId}/cancel`, "cancel");
    if (action === "advisor" && selectedJobId) postJobAction(`/api/jobs/${selectedJobId}/advisor/evaluate`, "advisor");
    if (action === "diagnostics") loadDeploymentStatus(true, true);
    if (action === "test_advisor") {
      try {
        await fetchJson("/api/advisor/test", { method: "POST" });
        setInfoMessage("AI 助手连接测试成功");
      } catch (e) {
        setErrorMessage("AI 连接测试失败");
      }
    }
  }

  async function postJobAction(path: string, key: string) {
    setActionKey(key);
    try {
      const payload = await fetchJson<JobPayload>(path, { method: "POST" });
      setInfoMessage(buildActionMessage(key, payload.job.job_id));
      if (selectedJobId === payload.job.job_id) {
        loadInspection(selectedJobId, false);
      }
      loadJobs(false);
    } catch (e) {
      setErrorMessage("执行操作失败");
    } finally {
      setActionKey(null);
    }
  }

  async function handleCreateJob() {
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
    files.forEach(f => formData.append("files", f, f.name));

    try {
      const payload = await fetchJson<JobPayload>("/api/jobs", { method: "POST", body: formData });
      setFiles([]);
      handleInspectJob(payload.job.job_id);
      setInfoMessage(`任务 ${payload.job.job_id} 已创建。`);
    } catch (e) {
      setErrorMessage(friendlyError(e, "创建任务失败"));
    } finally {
      setSubmitting(false);
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
    setFormState(prev => ({ ...prev, model: value, params: {} }));
    if (!modelContracts[value]) {
      try {
        const contract = await fetchJson<ModelContract>(`/api/models/${value}/contract`);
        setAppState(prev => prev ? { ...prev, modelContracts: { ...prev.modelContracts, [value]: contract } } : null);
        setFormState(prev => ({
          ...prev,
          source_type: contract.allowedSourceTypes[0] || "images",
          params: contract.paramSchema.fields.reduce((acc, f) => ({ ...acc, [f.key]: f.default }), {})
        }));
      } catch (e) {
        setErrorMessage("加载模型契约失败");
      }
    }
  }

  return (
    <div className="app-shell">
      <Sidebar activeWorkspace={activeWorkspace} onWorkspaceChange={handleWorkspaceChange} summary={summary} />
      
      <div className="main-view">
        <CommandBar 
          activeWorkspace={activeWorkspace} 
          serviceState={serviceState} 
          serviceStatusLabel={serviceStatusLabel(serviceState)}
          onRefresh={() => { loadJobs(); refreshAppState(); }}
          onCreateJob={() => handleWorkspaceChange("create")}
          onAction={handleCommandAction}
          title={workspaceTitle || ""}
        />

        <main className="workspace-content">
          {infoMessage && <MessageBanner kind="info" message={infoMessage} />}
          {errorMessage && <MessageBanner kind="error" message={errorMessage} />}

          {activeWorkspace === "queue" && (
            <QueueWorkspace 
              jobs={jobs} 
              modelCatalog={modelCatalog} 
              selectedJobId={selectedJobId} 
              onSelectJob={setSelectedJobId} 
              onInspectJob={handleInspectJob}
              onDispatchJob={async (id) => {
                try {
                  await fetchJson(`/api/jobs/${id}/dispatch`, { method: "POST" });
                  loadJobs();
                  setInfoMessage("任务已派发");
                } catch (e) {
                  setErrorMessage(friendlyError(e, "派发失败"));
                }
              }}
              onCancelJob={async (id) => {
                try {
                  await fetchJson(`/api/jobs/${id}/cancel`, { method: "POST" });
                  loadJobs();
                  setInfoMessage("任务已取消");
                } catch (e) {
                  setErrorMessage(friendlyError(e, "取消失败"));
                }
              }}
            />
          )}

          {activeWorkspace === "create" && (
            <div className="create-workspace">
              {/* Step 1: Mode & Model Selection */}
              <div className="create-section">
                <div className="create-section-header">
                  <span className="create-step-num">1</span>
                  <span className="create-section-title">选择模式</span>
                </div>
                <div className="create-mode-toggle">
                  <button
                    type="button"
                    className={`mode-toggle-btn ${createMode === "single" ? "active" : ""}`}
                    onClick={() => setCreateMode("single")}
                  >
                    单任务创建
                  </button>
                  <button
                    type="button"
                    className={`mode-toggle-btn ${createMode === "batch" ? "active" : ""}`}
                    onClick={() => setCreateMode("batch")}
                  >
                    多模型对比
                  </button>
                </div>
              </div>

              <div className="create-main-grid">
                {/* Left: Model Config */}
                <div className="create-left-panel">
                  <div className="create-section">
                    <div className="create-section-header">
                      <span className="create-step-num">2</span>
                      <span className="create-section-title">{createMode === "single" ? "选择模型" : "选择对比模型"}</span>
                    </div>
                    {createMode === "single" ? (
                      <div className="model-selector-grid">
                        {runnableModelCatalog.map((m) => (
                          <div
                            key={m.value}
                            className={`model-selector-card ${formState.model === m.value ? "selected" : ""}`}
                            onClick={() => updateModel(m.value)}
                          >
                            <div className="model-selector-header">
                              <strong>{m.label}</strong>
                              {formState.model === m.value && <span className="model-selected-badge">已选</span>}
                            </div>
                            <p className="model-selector-desc">{m.description}</p>
                            <div className="model-selector-meta">
                              <span>{m.source_types?.join(" / ") || "images"}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="batch-model-grid">
                        {batchCompatibleModels.map((m) => {
                          const hint = batchModelHints[m.value];
                          const selected = selectedModels.includes(m.value);
                          return (
                            <div
                              key={m.value}
                              className={`model-selector-card compact ${selected ? "selected" : ""} ${hint ? "blocked" : ""}`}
                              onClick={() => !hint && toggleBatchModel(m.value)}
                            >
                              <div className="model-selector-header">
                                <input
                                  type="checkbox"
                                  checked={selected}
                                  onChange={() => toggleBatchModel(m.value)}
                                  disabled={!!hint && !selected}
                                  onClick={(e) => e.stopPropagation()}
                                />
                                <strong>{m.label}</strong>
                              </div>
                              {hint && <small className="model-blocker-hint">{hint}</small>}
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>

                  {createMode === "single" && (
                    <div className="create-section">
                      <div className="create-section-header">
                        <span className="create-step-num">3</span>
                        <span className="create-section-title">参数配置</span>
                      </div>
                      <div className="create-config-form">
                        <label className="field compact">
                          <span>输入类型</span>
                          <select value={formState.source_type} onChange={(e) => updateFormField("source_type", e.target.value)}>
                            {appState?.sourceTypes.filter(st => selectedModelSourceTypes.includes(st.value)).map((st) => (
                              <option key={st.value} value={st.value}>{st.label}</option>
                            ))}
                          </select>
                        </label>
                        {selectedModelContract && selectedModelContract.paramSchema.fields.length > 0 && (
                          <DynamicParamForm fields={selectedModelContract.paramSchema.fields} values={formState.params} onChange={updateFormField} />
                        )}
                      </div>
                    </div>
                  )}

                  {createMode === "batch" && (
                    <div className="create-section">
                      <div className="create-section-header">
                        <span className="create-step-num">3</span>
                        <span className="create-section-title">输入配置</span>
                      </div>
                      <label className="field compact">
                        <span>输入类型</span>
                        <select value={formState.source_type} onChange={(e) => updateFormField("source_type", e.target.value)}>
                          {appState?.sourceTypes.map((st) => (
                            <option key={st.value} value={st.value}>{st.label}</option>
                          ))}
                        </select>
                      </label>
                    </div>
                  )}
                </div>

                {/* Right: File Upload & Submit */}
                <div className="create-right-panel">
                  <div className="create-section">
                    <div className="create-section-header">
                      <span className="create-step-num">{createMode === "single" ? "4" : "4"}</span>
                      <span className="create-section-title">上传文件</span>
                      {files.length > 0 && <span className="create-file-count">{files.length} 个文件</span>}
                    </div>
                    <label className="dropzone-pro">
                      <input type="file" multiple onChange={(e) => setFiles(Array.from(e.target.files ?? []))} />
                      <div className="dropzone-content">
                        <div className="dropzone-icon">📁</div>
                        <strong>点击选择文件或拖拽到此处</strong>
                        <span>支持图片、视频等格式</span>
                      </div>
                    </label>
                    {files.length > 0 && (
                      <div className="file-list">
                        {files.slice(0, 6).map((file) => (
                          <div className="file-item" key={`${file.name}-${file.size}`}>
                            <span className="file-name">{file.name}</span>
                            <span className="file-size">{formatFileSize(file.size)}</span>
                            <button className="file-remove" type="button" onClick={() => setFiles(prev => prev.filter(f => f !== file))}>×</button>
                          </div>
                        ))}
                        {files.length > 6 && <div className="file-item more">+{files.length - 6} 个文件</div>}
                      </div>
                    )}
                  </div>

                  <div className="create-section">
                    <label className="field compact">
                      <span>备注（可选）</span>
                      <input 
                        type="text" 
                        value={formState.notes} 
                        onChange={(e) => updateFormField("notes", e.target.value)} 
                        placeholder="实验目的或说明..."
                      />
                    </label>
                  </div>

                  <div className="create-submit-section">
                    {createMode === "single" ? (
                      <button
                        className="create-submit-btn primary"
                        type="button"
                        onClick={() => handleCreateJob()}
                        disabled={submitting || !!selectedModelLaunchBlocker || files.length === 0}
                      >
                        {submitting ? "创建中..." : "创建任务"}
                      </button>
                    ) : (
                      <div className="create-submit-group">
                        <button
                          className="create-submit-btn"
                          type="button"
                          onClick={() => handleBatchCompare(false)}
                          disabled={batchSubmitting || selectedModels.length < 1 || files.length === 0 || batchHasBlockers}
                        >
                          {batchSubmitting ? "创建中..." : "仅创建"}
                        </button>
                        <button
                          className="create-submit-btn primary"
                          type="button"
                          onClick={() => handleBatchCompare(true)}
                          disabled={batchSubmitting || selectedModels.length < 1 || files.length === 0 || batchHasBlockers}
                        >
                          {batchSubmitting ? "创建中..." : "创建并派发"}
                        </button>
                      </div>
                    )}
                    {selectedModelLaunchBlocker && (
                      <p className="create-blocker-msg">{selectedModelLaunchBlocker}</p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeWorkspace === "inspect" && selectedInspection && (
            <InspectWorkspace
              inspection={selectedInspection}
              advisorState={advisorState}
              savingEvaluation={savingEvaluation}
              onSaveEvaluation={async (id, p) => { 
                setSavingEvaluation(true); 
                try { 
                  await fetchJson(`/api/jobs/${id}/evaluation`, { method: "POST", body: JSON.stringify(p) }); 
                  loadInspection(id); 
                } finally { 
                  setSavingEvaluation(false); 
                } 
              }}
              onConfigureAdvisor={openAdvisorSettings}
              onAction={postJobAction}
              onPreviewAsset={setPreviewAsset}
              onOpenOutput={async (p) => { 
                const f = new FormData(); 
                f.append("relative_path", p); 
                await fetchJson(`/api/jobs/${selectedInspection.job.job_id}/open-output`, { method: "POST", body: f }); 
              }}
              assetUrl={(p) => `${API_BASE}${p.startsWith("/") ? "" : "/"}${p}`}
              modelCatalog={modelCatalog}
            />
          )}

          {activeWorkspace === "samples" && (
            <SampleMatrixPanel
              samplesPayload={samplesPayload}
              errorMessage={samplesError}
              modelCatalog={modelCatalog}
              onLocateJob={handleInspectJob}
              onCopy={async (v, l) => { await navigator.clipboard.writeText(v); setInfoMessage(`${l}已复制`); }}
            />
          )}

          {activeWorkspace === "compare" && (
            <CompareBoard
              sampleId={compareSampleId}
              packet={comparePacket}
              loading={compareLoading}
              error={compareError}
              modelCatalog={modelCatalog}
              apiBase={API_BASE}
              onInspectJob={handleInspectJob}
              onRefresh={() => compareSampleId && loadComparePacket(compareSampleId)}
              onPreviewAsset={(asset) => setPreviewAsset(asset as PreviewAsset)}
            />
          )}

          {activeWorkspace === "development" && (
            <section className="overview-development-grid">
              <DevelopmentCyclePanel items={developmentLanes} />
              <ResearchAccelerationPanel items={developmentLanes} />
            </section>
          )}

          {activeWorkspace === "system" && (
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
              activeModelCatalog={modelCatalog.filter(m => m.active_track)}
              deferredModelCatalog={modelCatalog.filter(m => !m.active_track)}
              samplesPayload={samplesPayload}
              samplesError={samplesError}
              modelCatalog={modelCatalog}
              openWorkspace={(w: string, id?: string) => { if (id) handleInspectJob(id); else handleWorkspaceChange(w as any); }}
              copyText={async (v: string, l: string) => { await navigator.clipboard.writeText(v); setInfoMessage(`${l}已复制`); }}
              serviceMessage={serviceMessage}
              backendStatus={backendStatus}
            />
          )}
        </main>
      </div>

      {advisorModalOpen && (
        <div className="settings-modal-backdrop" onClick={() => setAdvisorModalOpen(false)}>
          <div className="settings-modal" onClick={e => e.stopPropagation()}>
            <div className="preview-modal-head">
              <strong>Advisor 配置</strong>
              <button className="ghost-button small" onClick={() => setAdvisorModalOpen(false)}>关闭</button>
            </div>
            <form className="form-stack settings-form" onSubmit={saveAdvisorSettings}>
              <label className="field"><span>启用</span><input type="checkbox" checked={advisorForm.enabled} onChange={e => setAdvisorForm(c => ({...c, enabled: e.target.checked}))} /></label>
              <label className="field"><span>Provider</span><select value={advisorForm.model} onChange={e => setAdvisorForm(c => ({...c, model: e.target.value}))}>{advisorProviders.map(p => <optgroup key={p.id} label={p.label}>{p.models.map(m => <option key={m} value={m}>{m}</option>)}</optgroup>)}</select></label>
              <label className="field"><span>Base URL</span><input value={advisorForm.baseUrl} onChange={e => setAdvisorForm(c => ({...c, baseUrl: e.target.value}))} /></label>
              <label className="field"><span>API Key</span><input type="password" value={advisorForm.apiKey} onChange={e => setAdvisorForm(c => ({...c, apiKey: e.target.value}))} placeholder={advisorForm.hasApiKey ? "已保存" : "输入 Key"} /></label>
              <div className="settings-modal-actions"><button className="primary-button" type="submit">保存</button></div>
            </form>
          </div>
        </div>
      )}

      {previewAsset && (
        <div className="preview-modal-backdrop" onClick={() => setPreviewAsset(null)}>
          <div className="preview-modal" onClick={e => e.stopPropagation()}>
            <div className="preview-modal-head">
              <strong>{previewAsset.name}</strong>
              <button className="ghost-button small" onClick={() => setPreviewAsset(null)}>关闭</button>
            </div>
            <div className="preview-modal-body">
              {previewAsset.kind === "image" ? <img src={previewAsset.url} alt={previewAsset.name} /> : <video src={previewAsset.url} controls autoPlay />}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function SystemWorkbench(props: any) {
  const deploymentRows = buildDeploymentComponentRows(props.deploymentStatus, props.modelCatalog);
  return (
    <section className="system-grid workbench-system-grid">
      <article className="panel">
        <PanelTitle eyebrow="System" title="本地服务" />
        <div className="support-checklist compact-stack">
          <article className="support-check-item"><strong>状态</strong><p>{props.serviceMessage}</p></article>
          <article className="support-check-item"><strong>后端</strong><p>{backendStatusText(props.backendStatus)}</p></article>
        </div>
      </article>
      <article className="panel">
        <PanelTitle eyebrow="Advisor" title="AI 配置" />
        <div className="support-checklist compact-stack">
          <article className="support-check-item"><strong>状态</strong><p>{props.advisorReady ? `就绪：${props.advisorState.model}` : "未配置"}</p></article>
          <button className="ghost-button small" onClick={props.openAdvisorSettings}>打开配置</button>
        </div>
      </article>
      <article className="panel deployment-console-panel">
        <PanelTitle eyebrow="Deployment" title="部署状态" />
        <div className="deployment-readiness-table">
          <div className="deployment-readiness-head"><span>模型</span><span>目录</span><span>环境</span><span>文件</span><span>权重</span></div>
          {deploymentRows.map((row: any) => (
            <div className={`deployment-readiness-row ${row.tone}`} key={row.component}>
              <div className="deployment-model-cell">
                <strong>{modelDisplayName(row.component, props.modelCatalog)}</strong>
              </div>
              <span>{row.directory}</span><span>{row.env}</span><span>{row.files}</span><span>{row.checkpoints}</span>
            </div>
          ))}
        </div>
        <button className="ghost-button small" onClick={() => props.loadDeploymentStatus(true, true)}>立即刷新</button>
      </article>
    </section>
  );
}

export default App;
