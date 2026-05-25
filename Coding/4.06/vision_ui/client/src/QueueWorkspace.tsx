import { useMemo } from "react";
import { JobListItem } from "./workflowHelpers";
import { formatDateTime, modelDisplayName, sourceTypeLabel, statusLabel } from "./displayHelpers";
import { StatusBadge } from "./uiPrimitives";
import { ModelCatalogItem } from "./types";
import { 
  Play,
  Clock,
  CheckCircle2,
  AlertCircle,
  Loader2
} from "lucide-react";

interface QueueWorkspaceProps {
  jobs: JobListItem[];
  modelCatalog: ModelCatalogItem[];
  selectedJobId: string | null;
  onSelectJob: (jobId: string) => void;
  onInspectJob: (jobId: string) => void;
  onDispatchJob?: (jobId: string) => void;
  onRetryJob?: (jobId: string) => void;
  onCancelJob?: (jobId: string) => void;
}

export function QueueWorkspace({ 
  jobs, 
  modelCatalog, 
  selectedJobId, 
  onSelectJob, 
  onInspectJob,
  onDispatchJob,
  onRetryJob,
  onCancelJob
}: QueueWorkspaceProps) {
  const stats = useMemo(() => {
    const created = jobs.filter(j => j.job.status === "created").length;
    const running = jobs.filter(j => j.job.status === "running").length;
    const finished = jobs.filter(j => j.job.status === "finished").length;
    const failed = jobs.filter(j => j.job.status === "failed" || j.job.status === "cancelled").length;
    return { total: jobs.length, created, running, finished, failed };
  }, [jobs]);

  const runningJobs = useMemo(() => jobs.filter(j => j.job.status === "running"), [jobs]);
  const pendingJobs = useMemo(() => jobs.filter(j => j.job.status === "created"), [jobs]);
  const recentJobs = useMemo(() => jobs.slice(0, 10), [jobs]);

  return (
    <div className="queue-workspace">
      {/* Stats Bar */}
      <div className="queue-stats-bar">
        <div className="queue-stat">
          <Clock size={14} />
          <span className="queue-stat-value">{stats.created}</span>
          <span className="queue-stat-label">待派发</span>
        </div>
        <div className="queue-stat running">
          <Loader2 size={14} className="spin" />
          <span className="queue-stat-value">{stats.running}</span>
          <span className="queue-stat-label">运行中</span>
        </div>
        <div className="queue-stat success">
          <CheckCircle2 size={14} />
          <span className="queue-stat-value">{stats.finished}</span>
          <span className="queue-stat-label">已完成</span>
        </div>
        <div className="queue-stat danger">
          <AlertCircle size={14} />
          <span className="queue-stat-value">{stats.failed}</span>
          <span className="queue-stat-label">需处理</span>
        </div>
      </div>

      <div className="queue-main-grid">
        {/* Left: Running & Pending */}
        <div className="queue-lanes">
          {/* Running Jobs */}
          <div className="queue-lane">
            <div className="queue-lane-header">
              <span className="queue-lane-title">运行中</span>
              <span className="queue-lane-count">{runningJobs.length}</span>
            </div>
            <div className="queue-lane-list">
              {runningJobs.length > 0 ? runningJobs.map(item => (
                <div 
                  key={item.job.job_id} 
                  className={`queue-job-card running ${selectedJobId === item.job.job_id ? "selected" : ""}`}
                  onClick={() => onSelectJob(item.job.job_id)}
                >
                  <div className="queue-job-header">
                    <strong>{modelDisplayName(item.job.model, modelCatalog)}</strong>
                    <span className="queue-job-progress">{item.phase_display.percent}%</span>
                  </div>
                  <div className="queue-job-meta">
                    <code>{item.job.job_id.slice(0, 8)}</code>
                    <span>{item.phase_display.label}</span>
                  </div>
                  <div className="queue-job-progress-bar">
                    <div className="queue-job-progress-fill" style={{ width: `${item.phase_display.percent}%` }} />
                  </div>
                  <div className="queue-job-actions">
                    <button className="icon-btn" title="检视" onClick={(e) => { e.stopPropagation(); onInspectJob(item.job.job_id); }}>
                      检视
                    </button>
                    {onCancelJob && (
                      <button className="icon-btn danger" title="取消" onClick={(e) => { e.stopPropagation(); onCancelJob(item.job.job_id); }}>
                        取消
                      </button>
                    )}
                  </div>
                </div>
              )) : (
                <div className="queue-lane-empty">无运行中任务</div>
              )}
            </div>
          </div>

          {/* Pending Jobs */}
          <div className="queue-lane">
            <div className="queue-lane-header">
              <span className="queue-lane-title">待派发</span>
              <span className="queue-lane-count">{pendingJobs.length}</span>
            </div>
            <div className="queue-lane-list">
              {pendingJobs.length > 0 ? pendingJobs.map(item => (
                <div 
                  key={item.job.job_id} 
                  className={`queue-job-card pending ${selectedJobId === item.job.job_id ? "selected" : ""}`}
                  onClick={() => onSelectJob(item.job.job_id)}
                >
                  <div className="queue-job-header">
                    <strong>{modelDisplayName(item.job.model, modelCatalog)}</strong>
                    <StatusBadge state={item.job.status} label={statusLabel(item.job.status)} />
                  </div>
                  <div className="queue-job-meta">
                    <code>{item.job.job_id.slice(0, 8)}</code>
                    <span>{sourceTypeLabel(item.job.source_type)}</span>
                  </div>
                  <div className="queue-job-actions">
                    {onDispatchJob && (
                      <button className="icon-btn primary" title="派发" onClick={(e) => { e.stopPropagation(); onDispatchJob(item.job.job_id); }}>
                        <Play size={12} /> 派发
                      </button>
                    )}
                    <button className="icon-btn" title="检视" onClick={(e) => { e.stopPropagation(); onInspectJob(item.job.job_id); }}>
                      检视
                    </button>
                  </div>
                </div>
              )) : (
                <div className="queue-lane-empty">无待派发任务</div>
              )}
            </div>
          </div>
        </div>

        {/* Right: All Jobs Table */}
        <div className="queue-table-section">
          <div className="queue-table-header">
            <span className="queue-table-title">全部任务</span>
            <span className="queue-table-count">{jobs.length} 条记录</span>
          </div>
          <div className="workbench-table-container compact">
            <table className="workbench-table">
              <thead>
                <tr>
                  <th style={{width: "70px"}}>状态</th>
                  <th style={{width: "90px"}}>ID</th>
                  <th>模型</th>
                  <th style={{width: "70px"}}>来源</th>
                  <th style={{width: "100px"}}>时间</th>
                  <th style={{width: "80px"}}>进度</th>
                  <th style={{width: "60px"}}></th>
                </tr>
              </thead>
              <tbody>
                {recentJobs.length > 0 ? recentJobs.map((item) => (
                  <tr 
                    key={item.job.job_id} 
                    className={selectedJobId === item.job.job_id ? "selected" : ""}
                    onClick={() => onSelectJob(item.job.job_id)}
                  >
                    <td>
                      <StatusBadge state={item.job.status} label={statusLabel(item.job.status)} />
                    </td>
                    <td><code className="job-id-cell">{item.job.job_id.slice(0, 8)}</code></td>
                    <td><strong className="model-cell">{modelDisplayName(item.job.model, modelCatalog)}</strong></td>
                    <td>{sourceTypeLabel(item.job.source_type)}</td>
                    <td className="time-cell">{formatDateTime(item.job.created_at)}</td>
                    <td>
                      <span className="progress-cell">{item.phase_display.percent}%</span>
                    </td>
                    <td>
                      <button className="icon-btn" onClick={(e) => { e.stopPropagation(); onInspectJob(item.job.job_id); }}>
                        →
                      </button>
                    </td>
                  </tr>
                )) : (
                  <tr>
                    <td colSpan={7}>
                      <div className="queue-table-empty">暂无任务记录</div>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
