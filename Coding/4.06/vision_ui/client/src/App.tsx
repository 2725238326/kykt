import { FormEvent, useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type {
  BackendStatusPayload,
  BootstrapPayload,
  JobPayload,
  JobsListPayload,
  ResultSummary
} from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000").replace(/\/$/, "");

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
  num_frames: "80",
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

function App() {
  const [bootstrap, setBootstrap] = useState<BootstrapPayload | null>(null);
  const [jobs, setJobs] = useState<JobsListPayload["jobs"]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<JobPayload | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [actionKey, setActionKey] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [infoMessage, setInfoMessage] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<BackendStatusPayload | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [formState, setFormState] = useState<FormState>({
    model: "dust3r",
    source_type: "images",
    notes: "",
    ...defaultDust3rParams,
    ...defaultMonst3rParams
  });

  useEffect(() => {
    void loadBootstrap();
    void loadJobs();
    void loadDesktopBackendStatus();
    const timer = window.setInterval(() => void loadJobs(false), 4000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!selectedJobId && jobs[0]) {
      setSelectedJobId(jobs[0].job.job_id);
    }
  }, [jobs, selectedJobId]);

  useEffect(() => {
    if (!selectedJobId) {
      setSelectedJob(null);
      return;
    }
    void loadJobDetail(selectedJobId);
    const timer = window.setInterval(() => void loadJobDetail(selectedJobId, false), 4000);
    return () => window.clearInterval(timer);
  }, [selectedJobId]);

  const summary = bootstrap?.summary ?? {
    total: jobs.length,
    running: jobs.filter((item) => item.job.status === "running").length,
    finished: jobs.filter((item) => item.job.status === "finished").length,
    failed: jobs.filter((item) => item.job.status === "failed").length,
    cancelled: jobs.filter((item) => item.job.status === "cancelled").length
  };

  const selectedModel = useMemo(
    () => bootstrap?.models.find((item) => item.value === formState.model),
    [bootstrap?.models, formState.model]
  );

  const isDust3r = formState.model === "dust3r";
  const isMonst3r = formState.model === "monst3r";
  const selectedFileCount = files.length;
  const runningSelectedJob = selectedJob?.job.status === "running";
  const canDispatchSelectedJob = selectedJob
    ? selectedJob.job.status === "draft" ||
      selectedJob.job.status === "ready" ||
      selectedJob.job.status === "failed" ||
      selectedJob.job.status === "cancelled"
    : false;

  async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE}${path}`, init);
    if (!response.ok) {
      let message = `Request failed: ${response.status}`;
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

  async function loadBootstrap(showError = true) {
    try {
      setBootstrap(await fetchJson<BootstrapPayload>("/api/bootstrap"));
    } catch (error) {
      if (showError) {
        setErrorMessage(error instanceof Error ? error.message : "加载初始化信息失败。");
      }
    }
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
        setErrorMessage(error instanceof Error ? error.message : "加载任务列表失败。");
      }
    }
  }

  async function loadJobDetail(jobId: string, showError = true) {
    try {
      setSelectedJob(await fetchJson<JobPayload>(`/api/jobs/${jobId}`));
    } catch (error) {
      if (showError) {
        setErrorMessage(error instanceof Error ? error.message : "加载任务详情失败。");
      }
    }
  }

  function updateFormField(key: keyof typeof formState, value: string) {
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

    if (files.length === 0) {
      setErrorMessage("请先选择输入文件。");
      return;
    }

    if (isDust3r && files.length < 2) {
      setErrorMessage("DUSt3R 至少需要两张图片才能创建任务。");
      return;
    }

    if (isMonst3r && formState.source_type === "video" && files.length !== 1) {
      setErrorMessage("MonST3R 视频模式请上传 1 个视频文件；如果是一组图片，请选择“帧序列”。");
      return;
    }

    if (isMonst3r && formState.source_type !== "video" && files.length < 2) {
      setErrorMessage("MonST3R 帧序列模式建议至少上传 2 张图片。");
      return;
    }

    setSubmitting(true);
    const formData = new FormData();
    formData.append("model", formState.model);
    formData.append("source_type", formState.source_type);
    formData.append("notes", formState.notes);

    if (isDust3r) {
      Object.keys(defaultDust3rParams).forEach((key) => {
        formData.append(key, formState[key as keyof typeof formState]);
      });
    } else if (isMonst3r) {
      Object.keys(defaultMonst3rParams).forEach((key) => {
        formData.append(key, formState[key as keyof typeof formState]);
      });
    }

    files.forEach((file) => formData.append("files", file, file.name));

    try {
      const payload = await fetchJson<JobPayload>("/api/jobs", { method: "POST", body: formData });
      setFiles([]);
      setSelectedJobId(payload.job.job_id);
      setSelectedJob(payload);
      setInfoMessage(`任务 ${payload.job.job_id} 已创建，本地输入已经缓存。`);
      await loadJobs(false);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "创建任务失败。");
    } finally {
      setSubmitting(false);
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
      setInfoMessage(buildActionMessage(key, payload.job.job_id));
      await loadJobs(false);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "执行任务操作失败。");
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
      setErrorMessage(error instanceof Error ? error.message : "打开本地产物失败。");
    } finally {
      setActionKey(null);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">KYKT Vision Client</div>
          <h1>把旧网页换成真正顺手的本地客户端。</h1>
          <p>
            新界面先用 React + TypeScript 重建工作流，再接进 Tauri 2 做桌面壳。视觉方向按
            Apple 风格收敛，重点放在任务创建、远端进度、结果查看和本地操作都在一个界面里完成。
          </p>
        </div>
        <div className="server-chip-group">
          <div className="server-chip">
            <span>远端主机</span>
            <strong>
              {bootstrap?.server.user}@{bootstrap?.server.host}
            </strong>
          </div>
          <div className="server-chip muted">
            <span>远端根目录</span>
            <strong>{bootstrap?.server.remote_root ?? "/hdd3/kykt26"}</strong>
          </div>
          <div className="server-chip muted">
            <span>客户端阶段</span>
            <strong>React Rebuild + Tauri Shell</strong>
          </div>
          <div className={`server-chip ${backendStatusClass(backendStatus)}`}>
            <span>本地后端</span>
            <strong>{backendStatusLabel(backendStatus)}</strong>
            <small>{backendStatusHint(backendStatus)}</small>
          </div>
        </div>
      </header>

      <main className="grid">
        <section className="panel hero">
          <div>
            <div className="eyebrow dark">Desktop-first rebuild</div>
            <h2>统一输入、SSH 调度、结果和归档。</h2>
            <p>
              这版不再沿着旧模板页面堆功能，而是直接按本地客户端的思路来组织：任务创建区像控制台，任务详情区像实验记录，产物区像桌面工作区。
            </p>
          </div>
          <div className="stat-grid">
            <MetricCard label="总任务" value={String(summary.total)} />
            <MetricCard label="运行中" value={String(summary.running)} />
            <MetricCard label="已完成" value={String(summary.finished)} />
            <MetricCard label="失败/取消" value={String(summary.failed + summary.cancelled)} />
          </div>
        </section>

        {infoMessage ? <MessageBanner kind="info" message={infoMessage} /> : null}
        {errorMessage ? <MessageBanner kind="error" message={errorMessage} /> : null}

        <section className="panel composer">
          <div className="section-head">
            <div>
              <div className="eyebrow">Create</div>
              <h3>新建任务</h3>
            </div>
            <span className="soft-tag">{selectedModel?.description ?? "本地创建，远端执行"}</span>
          </div>

          <form className="stack" onSubmit={handleCreateJob}>
            <div className="row-two">
              <label className="field">
                <span>模型</span>
                <select
                  value={formState.model}
                  onChange={(event) => updateModel(event.target.value)}
                >
                  {bootstrap?.models.map((item) => (
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
                  {bootstrap?.source_types.map((item) => (
                    <option key={item.value} value={item.value}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <label className="field">
              <span>任务说明</span>
              <textarea
                rows={4}
                value={formState.notes}
                onChange={(event) => updateFormField("notes", event.target.value)}
                placeholder="记录目标、数据说明、预期输出或后续想比较的参数。"
              />
            </label>

            {isDust3r ? (
              <>
                <div className="inline-callout">
                  DUSt3R 当前默认按多图也能跑的参数来建任务。双图和多图不再强行分开，只要求至少两张输入图。
                </div>
                <div className="param-grid">
                  {Object.keys(defaultDust3rParams).map((key) => (
                    <label className="field compact" key={key}>
                      <span>{formatParamLabel(key)}</span>
                      <input
                        value={formState[key as keyof typeof formState]}
                        onChange={(event) =>
                          setFormState((current) => ({ ...current, [key]: event.target.value }))
                        }
                      />
                    </label>
                  ))}
                </div>
              </>
            ) : (
              <>
                <div className="inline-callout">
                  MonST3R 已接入服务器端官方 demo。视频模式上传 1 个视频；帧序列模式上传同一场景的多张连续图片。
                </div>
                <div className="param-grid monst3r-param-grid">
                  <label className="field compact">
                    <span>{formatParamLabel("image_size")}</span>
                    <select
                      value={formState.image_size}
                      onChange={(event) => updateFormField("image_size", event.target.value)}
                    >
                      <option value="512">512（质量优先）</option>
                      <option value="224">224（更快更省显存）</option>
                    </select>
                  </label>
                  <label className="field compact">
                    <span>{formatParamLabel("num_frames")}</span>
                    <input
                      type="number"
                      min="1"
                      max="2000"
                      value={formState.num_frames}
                      onChange={(event) => updateFormField("num_frames", event.target.value)}
                    />
                  </label>
                  <label className="field compact">
                    <span>{formatParamLabel("fps")}</span>
                    <input
                      type="number"
                      min="0"
                      max="120"
                      value={formState.fps}
                      onChange={(event) => updateFormField("fps", event.target.value)}
                    />
                  </label>
                  <label className="field compact">
                    <span>{formatParamLabel("batch_size")}</span>
                    <input
                      type="number"
                      min="1"
                      max="16"
                      value={formState.batch_size}
                      onChange={(event) => updateFormField("batch_size", event.target.value)}
                    />
                  </label>
                  <label className="field compact">
                    <span>{formatParamLabel("not_batchify")}</span>
                    <select
                      value={formState.not_batchify}
                      onChange={(event) => updateFormField("not_batchify", event.target.value)}
                    >
                      <option value="true">开启（推荐，省显存）</option>
                      <option value="false">关闭（更激进）</option>
                    </select>
                  </label>
                  <label className="field compact">
                    <span>{formatParamLabel("real_time")}</span>
                    <select
                      value={formState.real_time}
                      onChange={(event) => updateFormField("real_time", event.target.value)}
                    >
                      <option value="false">关闭（默认，高质量输出）</option>
                      <option value="true">开启（实时路径）</option>
                    </select>
                  </label>
                  <label className="field compact">
                    <span>{formatParamLabel("window_wise")}</span>
                    <select
                      value={formState.window_wise}
                      onChange={(event) => updateFormField("window_wise", event.target.value)}
                    >
                      <option value="false">关闭（默认）</option>
                      <option value="true">开启（长序列）</option>
                    </select>
                  </label>
                  <label className="field compact">
                    <span>{formatParamLabel("window_size")}</span>
                    <input
                      type="number"
                      min="2"
                      max="500"
                      value={formState.window_size}
                      onChange={(event) => updateFormField("window_size", event.target.value)}
                    />
                  </label>
                  <label className="field compact">
                    <span>{formatParamLabel("window_overlap_ratio")}</span>
                    <input
                      type="number"
                      min="0"
                      max="0.95"
                      step="0.05"
                      value={formState.window_overlap_ratio}
                      onChange={(event) => updateFormField("window_overlap_ratio", event.target.value)}
                    />
                  </label>
                </div>
              </>
            )}

            <label className="dropzone">
              <input
                type="file"
                multiple
                onChange={(event) => setFiles(Array.from(event.target.files ?? []))}
              />
              <span>
                拖入文件或点击选择。内部命名会自动规范化，不需要你手动改文件名。
              </span>
            </label>

            <div className="selection-strip">
              <span className="soft-tag">已选 {selectedFileCount} 个文件</span>
              {isDust3r ? <span className="soft-tag">至少 2 张图</span> : null}
              {isMonst3r ? <span className="soft-tag">视频 1 个 / 帧序列至少 2 张</span> : null}
            </div>

            {files.length > 0 ? (
              <div className="chip-row">
                {files.map((file) => (
                  <button
                    key={`${file.name}-${file.size}`}
                    className="chip removable"
                    type="button"
                    onClick={() => removePendingFile(file.name, file.size)}
                  >
                    <span>{file.name}</span>
                    <span className="chip-close">×</span>
                  </button>
                ))}
              </div>
            ) : null}

            <button className="primary-button" disabled={submitting} type="submit">
              {submitting ? "正在创建..." : "创建本地任务"}
            </button>
          </form>
        </section>

        <section className="panel jobs">
          <div className="section-head">
            <div>
              <div className="eyebrow">Jobs</div>
              <h3>任务列表</h3>
            </div>
            <span className="soft-tag">每 4 秒自动刷新</span>
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
                  <div className="job-card-top">
                    <strong>{item.job.job_id}</strong>
                    <StatusPill status={item.job.status} />
                  </div>
                  <div className="job-model">{statusModelLabel(item.job.model)}</div>
                  <div className="job-note">{item.job.notes || "无备注"}</div>
                  <div className="job-meta-line">
                    <span>{sourceTypeLabel(item.job.source_type)}</span>
                    <span>{formatDateTime(item.job.created_at)}</span>
                  </div>
                  <div className="progress-track">
                    <div className="progress-fill" style={{ width: `${item.phase_display.percent}%` }} />
                  </div>
                  <small>{item.phase_display.label}</small>
                </button>
              ))
            ) : (
              <div className="empty-state">
                还没有任务。先上传一组图片或视频，新的客户端会把它整理成标准任务。
              </div>
            )}
          </div>
        </section>

        <section className="panel detail">
          <div className="section-head">
            <div>
              <div className="eyebrow">Detail</div>
              <h3>任务详情</h3>
            </div>
            {selectedJob ? <StatusPill status={selectedJob.job.status} /> : null}
          </div>

          {selectedJob ? (
            <div className="stack detail-stack">
              <div className="detail-hero">
                <div className="detail-copy">
                  <div className="job-id">{selectedJob.job.job_id}</div>
                  <div className="detail-phase">{selectedJob.phase_display.label}</div>
                  <p>{selectedJob.job.progress_message || selectedJob.phase_display.description}</p>
                  <div className="detail-meta-list">
                    <span>{statusModelLabel(selectedJob.job.model)}</span>
                    <span>{sourceTypeLabel(selectedJob.job.source_type)}</span>
                    <span>{formatDateTime(selectedJob.job.created_at)}</span>
                    {selectedJob.job.remote_runner ? <span>{selectedJob.job.remote_runner}</span> : null}
                  </div>
                </div>
                <div className="detail-score-block">
                  <div className="detail-score">{selectedJob.phase_display.percent}%</div>
                  <div className="soft-tag">
                    {selectedJob.previews.length} 输入 / {selectedJob.outputs.length} 产物
                  </div>
                </div>
              </div>

              <div className="action-row">
                <button
                  disabled={!canDispatchSelectedJob || actionKey === "dispatch"}
                  onClick={() => postJobAction(`/api/jobs/${selectedJob.job.job_id}/dispatch`, "dispatch")}
                  type="button"
                >
                  {actionKey === "dispatch" ? "正在启动..." : "运行"}
                </button>
                <button
                  disabled={runningSelectedJob || actionKey === "retry"}
                  onClick={() => postJobAction(`/api/jobs/${selectedJob.job.job_id}/retry`, "retry")}
                  type="button"
                >
                  {actionKey === "retry" ? "正在重试..." : "重试"}
                </button>
                <button
                  disabled={actionKey === "duplicate"}
                  onClick={() => postJobAction(`/api/jobs/${selectedJob.job.job_id}/duplicate`, "duplicate")}
                  type="button"
                >
                  {actionKey === "duplicate" ? "正在复制..." : "复制"}
                </button>
                <button
                  className="danger"
                  disabled={!runningSelectedJob || actionKey === "cancel"}
                  onClick={() => postJobAction(`/api/jobs/${selectedJob.job.job_id}/cancel`, "cancel")}
                  type="button"
                >
                  {actionKey === "cancel" ? "正在取消..." : "取消"}
                </button>
              </div>

              <div className="detail-grid">
                <article className="card-block">
                  <h4>任务概览</h4>
                  <SummaryPanel summary={selectedJob.result_summary} />
                </article>

                <article className="card-block">
                  <h4>输入预览</h4>
                  <div className="preview-grid">
                    {selectedJob.previews.length > 0 ? (
                      selectedJob.previews.map((preview) => (
                        <a
                          key={preview.relative_path}
                          className="preview-card"
                          href={assetUrl(preview.url)}
                          target="_blank"
                          rel="noreferrer"
                        >
                          {preview.is_image ? <img src={assetUrl(preview.url)} alt={preview.display_name} /> : null}
                          <span>{preview.display_name}</span>
                        </a>
                      ))
                    ) : (
                      <span className="muted-text">暂时没有输入预览。</span>
                    )}
                  </div>
                </article>
              </div>

              <div className="detail-grid">
                <article className="card-block">
                  <h4>输出产物</h4>
                  {selectedJob.outputs.length > 0 ? (
                    <div className="output-grid">
                      {selectedJob.outputs.map((output) => (
                        <article className="output-card" key={output.relative_path}>
                          {output.is_image ? (
                            <a href={assetUrl(output.url)} target="_blank" rel="noreferrer">
                              <img
                                className="output-preview"
                                src={assetUrl(output.url)}
                                alt={output.display_name}
                              />
                            </a>
                          ) : (
                            <div
                              className={`output-preview placeholder ${
                                output.is_pointcloud || output.is_model3d ? "pointcloud" : ""
                              }`}
                            >
                              {output.is_pointcloud
                                ? "PLY"
                                : output.is_model3d
                                  ? "GLB"
                                  : fileExtensionLabel(output.display_name)}
                            </div>
                          )}
                          <div className="output-body">
                            <strong>{output.display_name}</strong>
                            <span className="muted-text">{describeOutput(output.display_name)}</span>
                            <div className="output-actions">
                              <a href={assetUrl(output.url)} target="_blank" rel="noreferrer">
                                查看
                              </a>
                              <a href={assetUrl(output.url)} download>
                                下载
                              </a>
                              <button
                                disabled={actionKey === `open:${output.relative_path}`}
                                onClick={() => openOutput(output.relative_path)}
                                type="button"
                              >
                                {actionKey === `open:${output.relative_path}` ? "打开中..." : "本地打开"}
                              </button>
                            </div>
                          </div>
                        </article>
                      ))}
                    </div>
                  ) : (
                    <span className="muted-text">结果还没有回传到本地缓存。</span>
                  )}
                </article>

                <article className="card-block">
                  <h4>阶段与日志</h4>
                  <div className="timeline">
                    {selectedJob.phase_display.steps.map((step) => (
                      <div className={`timeline-step ${step.state}`} key={step.code}>
                        <strong>{step.label}</strong>
                        <span>{step.hint}</span>
                      </div>
                    ))}
                  </div>
                  <div className="log-list">
                    {selectedJob.logs.length > 0 ? (
                      selectedJob.logs.map((log) => (
                        <div className="log-card" key={log.relative_path}>
                          <strong>{log.name}</strong>
                          <pre>{log.tail || "暂无日志。"}</pre>
                        </div>
                      ))
                    ) : (
                      <span className="muted-text">还没有日志。</span>
                    )}
                  </div>
                </article>
              </div>
            </div>
          ) : (
            <div className="empty-state">先从左侧选择一个任务，或先创建一条新的任务。</div>
          )}
        </section>

        <section className="panel rebuild-notes">
          <div className="section-head">
            <div>
              <div className="eyebrow">Rebuild</div>
              <h3>当前还在推进的部分</h3>
            </div>
          </div>
          <div className="note-list">
            {bootstrap?.delivery_gaps.map((gap) => (
              <article className="note-card" key={gap.title}>
                <h4>{gap.title}</h4>
                <p>{gap.detail}</p>
              </article>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}

function MetricCard(props: { label: string; value: string }) {
  return (
    <div className="metric-card">
      <span>{props.label}</span>
      <strong>{props.value}</strong>
    </div>
  );
}

function MessageBanner(props: { kind: "info" | "error"; message: string }) {
  return (
    <section className={`panel message-panel ${props.kind}`}>
      <div className="message-title">{props.kind === "info" ? "提示" : "错误"}</div>
      <div>{props.message}</div>
    </section>
  );
}

function SummaryPanel(props: { summary: ResultSummary | null }) {
  if (!props.summary) {
    return <span className="muted-text">任务摘要会在远端结果回传后自动生成。</span>;
  }

  const highlights = props.summary.highlights ?? [];
  const nextActions = props.summary.next_actions ?? [];
  const sceneMeta = props.summary.scene_meta ?? {};

  return (
    <div className="summary-panel">
      <div className="summary-stats">
        <SummaryStat label="状态" value={props.summary.status_label} />
        <SummaryStat
          label="耗时"
          value={formatDuration(props.summary.duration_seconds ?? null)}
        />
        <SummaryStat
          label="输入数"
          value={String(props.summary.inputs?.count ?? props.summary.inputs?.names?.length ?? 0)}
        />
        <SummaryStat
          label="产物数"
          value={String(props.summary.artifacts?.length ?? 0)}
        />
      </div>

      {Object.keys(sceneMeta).length > 0 ? (
        <div className="summary-meta-grid">
          {sceneMeta.n_pairs !== undefined ? (
            <SummaryStat label="图像配对" value={String(sceneMeta.n_pairs)} compact />
          ) : null}
          {sceneMeta.n_points !== undefined ? (
            <SummaryStat label="最终点数" value={String(sceneMeta.n_points)} compact />
          ) : null}
          {sceneMeta.raw_point_count !== undefined ? (
            <SummaryStat label="原始点数" value={String(sceneMeta.raw_point_count)} compact />
          ) : null}
        </div>
      ) : null}

      {highlights.length > 0 ? (
        <div className="summary-block">
          <h5>关键结果</h5>
          <ul>
            {highlights.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      ) : null}

      {nextActions.length > 0 ? (
        <div className="summary-block">
          <h5>建议下一步</h5>
          <ul>
            {nextActions.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

function SummaryStat(props: { label: string; value: string; compact?: boolean }) {
  return (
    <div className={`summary-stat ${props.compact ? "compact" : ""}`}>
      <span>{props.label}</span>
      <strong>{props.value}</strong>
    </div>
  );
}

function StatusPill(props: { status: string }) {
  return <span className={`status-pill ${props.status}`}>{statusLabel(props.status)}</span>;
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
      return "已完成";
    case "failed":
      return "失败";
    case "cancelled":
      return "已取消";
    case "draft":
      return "草稿";
    case "ready":
      return "已就绪";
    default:
      return status;
  }
}

function backendStatusClass(status: BackendStatusPayload | null) {
  if (!status) {
    return "muted";
  }
  if (status.running) {
    return "healthy";
  }
  return "warning";
}

function backendStatusLabel(status: BackendStatusPayload | null) {
  if (!status) {
    return "浏览器预览";
  }
  if (status.running && status.managed_by_tauri) {
    return "桌面端已托管";
  }
  if (status.running) {
    return "已连接现有后端";
  }
  return "未启动";
}

function backendStatusHint(status: BackendStatusPayload | null) {
  if (!status) {
    return "网页模式下不读取 Tauri 状态";
  }
  if (status.running && status.log_path) {
    return `日志：${status.log_path}`;
  }
  if (status.backend_root) {
    return `根目录：${status.backend_root}`;
  }
  return status.message || "暂无状态信息";
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
      return "点云模型，可在 MeshLab 中进一步检查";
    case "glb":
    case "gltf":
      return "MonST3R 三维场景，可用 3D 查看器或 MeshLab 打开";
    case "txt":
      return "轨迹或相机内参文本，可用于复盘和报告";
    case "npy":
      return "数组产物，通常是深度、置信度或中间几何结果";
    case "mp4":
    case "mov":
    case "avi":
    case "mkv":
    case "webm":
      return "视频产物，可本地播放或归档展示";
    default:
      return "本地任务产物";
  }
}

function fileExtensionLabel(filename: string) {
  const suffix = filename.split(".").pop()?.toUpperCase();
  return suffix || "FILE";
}

function formatDateTime(value: string) {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  }).format(parsed);
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
