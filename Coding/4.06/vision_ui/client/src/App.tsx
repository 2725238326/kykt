import { FormEvent, useEffect, useMemo, useState } from "react";
import type { BootstrapPayload, JobPayload, JobsListPayload } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

const defaultDust3rParams = {
  image_size: "512",
  scene_graph: "complete",
  niter: "300",
  lr: "0.01",
  batch_size: "1",
  max_points: "250000",
  match_viz_count: "50"
};

function App() {
  const [bootstrap, setBootstrap] = useState<BootstrapPayload | null>(null);
  const [jobs, setJobs] = useState<JobsListPayload["jobs"]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<JobPayload | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [formState, setFormState] = useState({
    model: "dust3r",
    source_type: "images",
    notes: "",
    ...defaultDust3rParams
  });

  useEffect(() => {
    void loadBootstrap();
    void loadJobs();
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

  async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE}${path}`, init);
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `Request failed: ${response.status}`);
    }
    return (await response.json()) as T;
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

  async function handleCreateJob(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (files.length === 0) {
      setErrorMessage("请先选择输入文件。");
      return;
    }

    setSubmitting(true);
    setErrorMessage(null);
    const formData = new FormData();
    formData.append("model", formState.model);
    formData.append("source_type", formState.source_type);
    formData.append("notes", formState.notes);
    if (formState.model === "dust3r") {
      Object.keys(defaultDust3rParams).forEach((key) => {
        formData.append(key, formState[key as keyof typeof formState]);
      });
    }
    files.forEach((file) => formData.append("files", file, file.name));

    try {
      const payload = await fetchJson<JobPayload>("/api/jobs", { method: "POST", body: formData });
      setFiles([]);
      setSelectedJobId(payload.job.job_id);
      await Promise.all([loadJobs(false), loadJobDetail(payload.job.job_id, false)]);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "创建任务失败。");
    } finally {
      setSubmitting(false);
    }
  }

  async function postJobAction(path: string) {
    try {
      const payload = await fetchJson<JobPayload>(path, { method: "POST" });
      setSelectedJob(payload);
      setSelectedJobId(payload.job.job_id);
      await loadJobs(false);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "执行任务操作失败。");
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">KYKT Vision Client</div>
          <h1>全面重构版客户端</h1>
          <p>React + TypeScript 先接现有后端，后续直接包进 Tauri。</p>
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
        </div>
      </header>

      <main className="grid">
        <section className="panel hero">
          <div>
            <div className="eyebrow dark">Apple-inspired interface</div>
            <h2>把旧网页换成更像产品的本地实验台。</h2>
            <p>
              视觉基调按 Apple 的黑、灰、蓝体系来做，核心目标不是再补一个页面，而是把本地文件、SSH 调度、任务状态和结果查看统一到一个桌面式工作流里。
            </p>
          </div>
          <div className="stat-grid">
            <MetricCard label="总任务" value={String(summary.total)} />
            <MetricCard label="运行中" value={String(summary.running)} />
            <MetricCard label="已完成" value={String(summary.finished)} />
            <MetricCard label="失败/取消" value={String(summary.failed + summary.cancelled)} />
          </div>
        </section>

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
                  onChange={(event) =>
                    setFormState((current) => ({ ...current, model: event.target.value }))
                  }
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
                  onChange={(event) =>
                    setFormState((current) => ({ ...current, source_type: event.target.value }))
                  }
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
                onChange={(event) =>
                  setFormState((current) => ({ ...current, notes: event.target.value }))
                }
                placeholder="记录目标、数据说明、预期输出。"
              />
            </label>

            {formState.model === "dust3r" ? (
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
            ) : (
              <div className="monst3r-note">
                MonST3R 会在下一阶段补成“视频或帧序列一键上传 + 远端 demo 启动 + 结果回传”的完整桌面流。
              </div>
            )}

            <label className="dropzone">
              <input
                type="file"
                multiple
                onChange={(event) => setFiles(Array.from(event.target.files ?? []))}
              />
              <span>拖入文件或点击选择，系统会自动规范内部命名。</span>
            </label>

            {files.length > 0 ? (
              <div className="chip-row">
                {files.map((file) => (
                  <span className="chip" key={`${file.name}-${file.size}`}>
                    {file.name}
                  </span>
                ))}
              </div>
            ) : null}

            <button className="primary-button" disabled={submitting} type="submit">
              {submitting ? "正在创建..." : "创建本地任务"}
            </button>
          </form>

          {errorMessage ? <div className="error-banner">{errorMessage}</div> : null}
        </section>

        <section className="panel jobs">
          <div className="section-head">
            <div>
              <div className="eyebrow">Jobs</div>
              <h3>任务列表</h3>
            </div>
          </div>
          <div className="job-list">
            {jobs.map((item) => (
              <button
                key={item.job.job_id}
                className={`job-card ${selectedJobId === item.job.job_id ? "active" : ""}`}
                onClick={() => setSelectedJobId(item.job.job_id)}
              >
                <div className="job-card-top">
                  <strong>{item.job.job_id}</strong>
                  <StatusPill status={item.job.status} />
                </div>
                <div className="job-model">{item.job.model}</div>
                <div className="job-note">{item.job.notes || "无备注"}</div>
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: `${item.phase_display.percent}%` }} />
                </div>
                <small>{item.phase_display.label}</small>
              </button>
            ))}
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
                <div>
                  <div className="job-id">{selectedJob.job.job_id}</div>
                  <div className="detail-phase">{selectedJob.phase_display.label}</div>
                  <p>{selectedJob.job.progress_message || selectedJob.phase_display.description}</p>
                </div>
                <div className="detail-score">{selectedJob.phase_display.percent}%</div>
              </div>

              <div className="action-row">
                <button onClick={() => postJobAction(`/api/jobs/${selectedJob.job.job_id}/dispatch`)}>运行</button>
                <button onClick={() => postJobAction(`/api/jobs/${selectedJob.job.job_id}/retry`)}>重试</button>
                <button onClick={() => postJobAction(`/api/jobs/${selectedJob.job.job_id}/duplicate`)}>复制</button>
                <button className="danger" onClick={() => postJobAction(`/api/jobs/${selectedJob.job.job_id}/cancel`)}>
                  取消
                </button>
              </div>

              <div className="detail-grid">
                <article className="card-block">
                  <h4>输入</h4>
                  <div className="preview-grid">
                    {selectedJob.previews.map((preview) => (
                      <a key={preview.relative_path} className="preview-card" href={preview.url} target="_blank" rel="noreferrer">
                        {preview.is_image ? <img src={preview.url} alt={preview.display_name} /> : null}
                        <span>{preview.display_name}</span>
                      </a>
                    ))}
                  </div>
                </article>

                <article className="card-block">
                  <h4>输出</h4>
                  <div className="output-list">
                    {selectedJob.outputs.length > 0 ? (
                      selectedJob.outputs.map((output) => (
                        <a key={output.relative_path} href={output.url} target="_blank" rel="noreferrer">
                          {output.display_name}
                        </a>
                      ))
                    ) : (
                      <span className="muted-text">结果还没有回传。</span>
                    )}
                  </div>
                </article>
              </div>

              <div className="detail-grid">
                <article className="card-block">
                  <h4>阶段</h4>
                  <div className="timeline">
                    {selectedJob.phase_display.steps.map((step) => (
                      <div className={`timeline-step ${step.state}`} key={step.code}>
                        <strong>{step.label}</strong>
                        <span>{step.hint}</span>
                      </div>
                    ))}
                  </div>
                </article>

                <article className="card-block">
                  <h4>日志</h4>
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
            <div className="empty-state">先从左侧选择一个任务，或新建一条任务。</div>
          )}
        </section>

        <section className="panel rebuild-notes">
          <div className="section-head">
            <div>
              <div className="eyebrow">Notes</div>
              <h3>重构关注点</h3>
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

function StatusPill(props: { status: string }) {
  return <span className={`status-pill ${props.status}`}>{statusLabel(props.status)}</span>;
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
    default:
      return status;
  }
}

function formatParamLabel(key: string) {
  return key.replace(/_/g, " ").replace(/\b\w/g, (char: string) => char.toUpperCase());
}

export default App;
