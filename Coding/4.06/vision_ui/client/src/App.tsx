import { FormEvent, useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type {
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
  image_size: "224",
  batch_size: "1",
  fps: "0",
  num_frames: "24",
  not_batchify: "true",
  real_time: "false",
  window_wise: "false",
  window_size: "100",
  window_overlap_ratio: "0.5"
};

type FormState = {
  model: string;
  source_type: string;
  notes: string;
} & typeof defaultDust3rParams &
  typeof defaultMonst3rParams;

type ServiceState = "starting" | "ready" | "degraded";

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
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [infoMessage, setInfoMessage] = useState<string | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [formState, setFormState] = useState<FormState>({
    model: "dust3r",
    source_type: "images",
    notes: "",
    ...defaultDust3rParams,
    ...defaultMonst3rParams
  });

  const bootstrapData = bootstrap ?? DEFAULT_BOOTSTRAP;
  const serviceReady = serviceState === "ready";
  const isDust3r = formState.model === "dust3r";
  const isMonst3r = formState.model === "monst3r";
  const selectedFileCount = files.length;
  const selectedModel = useMemo(
    () => bootstrapData.models.find((item) => item.value === formState.model),
    [bootstrapData.models, formState.model]
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
      setBackendStatus(await invoke<BackendStatusPayload>("backend_status"));
    } catch {
      setBackendStatus(null);
    }
  }

  async function loadJobs(showError = true) {
    try {
      const payload = await fetchJson<JobsListPayload>("/api/jobs");
      setJobs(payload.jobs);
      setBootstrap((current) => (current ? { ...current, summary: payload.summary } : current));
    } catch (error) {
      if (showError) {
        setErrorMessage(friendlyError(error, "加载任务列表失败。"));
      }
    }
  }

  async function loadJobDetail(jobId: string, showError = true) {
    try {
      setSelectedJob(await fetchJson<JobPayload>(`/api/jobs/${jobId}`));
    } catch (error) {
      if (showError) {
        setErrorMessage(friendlyError(error, "加载任务详情失败。"));
      }
    }
  }

  function updateFormField(key: keyof FormState, value: string) {
    setFormState((current) => ({ ...current, [key]: value }));
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
        source_type: current.source_type === "video" ? "images" : current.source_type
      };
    });
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

    const paramDefaults = isDust3r ? defaultDust3rParams : defaultMonst3rParams;
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
    if (isDust3r && files.length < 2) {
      return "DUSt3R 至少需要两张图片。";
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
          <p>{backendStatusText(backendStatus)}</p>
        </section>

        {infoMessage ? <MessageBanner kind="info" message={infoMessage} /> : null}
        {errorMessage ? <MessageBanner kind="error" message={errorMessage} /> : null}

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
                    {bootstrapData.source_types.map((item) => (
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
                {isDust3r ? (
                  <div className="param-grid">
                    {Object.keys(defaultDust3rParams).map((key) => (
                      <ParamField
                        key={key}
                        name={key}
                        value={formState[key as keyof FormState]}
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
              <MiniStat label="运行" value={summary.running} />
              <MiniStat label="完成" value={summary.finished} />
              <MiniStat label="失败" value={summary.failed} />
            </div>
            <div className="job-list">
              {jobs.length > 0 ? (
                jobs.map((item) => (
                  <button
                    key={item.job.job_id}
                    className={`job-card ${selectedJobId === item.job.job_id ? "active" : ""}`}
                    onClick={() => setSelectedJobId(item.job.job_id)}
                    type="button"
                  >
                    <div>
                      <strong>{item.job.job_id}</strong>
                      <p>{item.job.notes || statusModelLabel(item.job.model)}</p>
                    </div>
                    <StatusBadge state={item.job.status} label={statusLabel(item.job.status)} />
                    <div className="progress-track">
                      <div className="progress-fill" style={{ width: `${item.phase_display.percent}%` }} />
                    </div>
                  </button>
                ))
              ) : (
                <div className="empty-state">
                  暂无任务。先在左侧选择输入文件，创建第一条任务。
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
            />
          ) : (
            <div className="empty-state large">
              这里会显示任务进度、输入预览、输出结果和日志。页面先保持干净，不再一打开就塞满信息。
            </div>
          )}
        </section>
      </main>
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
}) {
  const job = props.selectedJob.job;
  return (
    <div className="detail-stack">
      <div className="detail-topline">
        <div>
          <StatusBadge state={job.status} label={statusLabel(job.status)} />
          <h3>{props.selectedJob.phase_display.label}</h3>
          <p>{job.progress_message || props.selectedJob.phase_display.description}</p>
        </div>
        <div className="detail-score">{props.selectedJob.phase_display.percent}%</div>
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
      </div>

      <div className="result-grid">
        <article className="soft-panel">
          <h4>结果摘要</h4>
          <SummaryPanel summary={props.selectedJob.result_summary} />
        </article>
        <article className="soft-panel">
          <h4>输入</h4>
          <div className="preview-grid">
            {props.selectedJob.previews.length > 0 ? (
              props.selectedJob.previews.map((preview) => (
                <a
                  key={preview.relative_path}
                  className="preview-card"
                  href={props.assetUrl(preview.url)}
                  target="_blank"
                  rel="noreferrer"
                >
                  {preview.is_image ? <img src={props.assetUrl(preview.url)} alt={preview.display_name} /> : null}
                  <span>{preview.display_name}</span>
                </a>
              ))
            ) : (
              <span className="muted-text">暂无输入预览。</span>
            )}
          </div>
        </article>
      </div>

      <article className="soft-panel">
        <h4>输出</h4>
        {props.selectedJob.outputs.length > 0 ? (
          <div className="output-grid">
            {props.selectedJob.outputs.map((output) => (
              <article className="output-card" key={output.relative_path}>
                {output.is_image ? (
                  <a href={props.assetUrl(output.url)} target="_blank" rel="noreferrer">
                    <img className="output-preview" src={props.assetUrl(output.url)} alt={output.display_name} />
                  </a>
                ) : (
                  <div className="output-preview placeholder">
                    {output.is_pointcloud ? "PLY" : output.is_model3d ? "GLB" : fileExtensionLabel(output.display_name)}
                  </div>
                )}
                <div>
                  <strong>{output.display_name}</strong>
                  <p>{describeOutput(output.display_name)}</p>
                  <div className="output-actions">
                    <a href={props.assetUrl(output.url)} target="_blank" rel="noreferrer">
                      查看
                    </a>
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
        ) : (
          <span className="muted-text">结果回传后会出现在这里。</span>
        )}
      </article>

      <article className="soft-panel">
        <h4>日志</h4>
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

function Monst3rParams(props: {
  formState: FormState;
  updateFormField: (key: keyof FormState, value: string) => void;
}) {
  return (
    <div className="param-grid">
      <label className="field compact">
        <span>{formatParamLabel("image_size")}</span>
        <select
          value={props.formState.image_size}
          onChange={(event) => props.updateFormField("image_size", event.target.value)}
        >
          <option value="512">512（质量优先）</option>
          <option value="224">224（更快）</option>
        </select>
      </label>
      {Object.keys(defaultMonst3rParams)
        .filter((key) => key !== "image_size")
        .map((key) => (
          <ParamField
            key={key}
            name={key}
            value={props.formState[key as keyof FormState]}
            onChange={(value) => props.updateFormField(key as keyof FormState, value)}
          />
        ))}
    </div>
  );
}

function ParamField(props: { name: string; value: string; onChange: (value: string) => void }) {
  if (["not_batchify", "real_time", "window_wise"].includes(props.name)) {
    return (
      <label className="field compact">
        <span>{formatParamLabel(props.name)}</span>
        <select value={props.value} onChange={(event) => props.onChange(event.target.value)}>
          <option value="true">开启</option>
          <option value="false">关闭</option>
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

function PanelTitle(props: { eyebrow: string; title: string }) {
  return (
    <div className="panel-title">
      <span>{props.eyebrow}</span>
      <h2>{props.title}</h2>
    </div>
  );
}

function MiniStat(props: { label: string; value: number }) {
  return (
    <div className="mini-stat">
      <span>{props.label}</span>
      <strong>{props.value}</strong>
    </div>
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

  return (
    <div className="summary-panel">
      <div className="summary-strip">
        <SummaryStat label="状态" value={props.summary.status_label} />
        <SummaryStat label="耗时" value={formatDuration(props.summary.duration_seconds ?? null)} />
        <SummaryStat label="输入" value={String(props.summary.inputs?.count ?? 0)} />
        <SummaryStat label="产物" value={String(props.summary.artifacts?.length ?? 0)} />
      </div>
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

function inputHint(model: string, sourceType: string) {
  if (model === "monst3r" && sourceType === "video") {
    return "上传 1 个视频文件";
  }
  if (model === "monst3r") {
    return "上传连续帧图片，建议 3 张以上";
  }
  return "上传 2 张或更多图片";
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
    case "monst3r":
      return "MonST3R";
    default:
      return model;
  }
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

export default App;
