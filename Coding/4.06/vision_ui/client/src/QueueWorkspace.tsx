import { JobListItem } from "./workflowHelpers";
import { formatDateTime, modelDisplayName, sourceTypeLabel, statusLabel } from "./displayHelpers";
import { StatusBadge, ModelSemanticChips } from "./uiPrimitives";
import { ModelCatalogItem } from "./types";

interface QueueWorkspaceProps {
  jobs: JobListItem[];
  modelCatalog: ModelCatalogItem[];
  selectedJobId: string | null;
  onSelectJob: (jobId: string) => void;
  onInspectJob: (jobId: string) => void;
}

export function QueueWorkspace({ jobs, modelCatalog, selectedJobId, onSelectJob, onInspectJob }: QueueWorkspaceProps) {
  return (
    <div className="workbench-table-container">
      <table className="workbench-table">
        <thead>
          <tr>
            <th>状态</th>
            <th>任务 ID</th>
            <th>模型</th>
            <th>来源</th>
            <th>创建时间</th>
            <th>进度</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          {jobs.length > 0 ? (
            jobs.map((item) => (
              <tr 
                key={item.job.job_id} 
                className={selectedJobId === item.job.job_id ? "selected" : ""}
                onClick={() => onSelectJob(item.job.job_id)}
              >
                <td>
                  <StatusBadge state={item.job.status} label={statusLabel(item.job.status)} />
                </td>
                <td>
                  <strong>{item.job.job_id}</strong>
                </td>
                <td>
                  <div className="badge-group">
                    <span>{modelDisplayName(item.job.model, modelCatalog)}</span>
                    <ModelSemanticChips catalog={modelCatalog} model={item.job.model} compact />
                  </div>
                </td>
                <td>{sourceTypeLabel(item.job.source_type)}</td>
                <td>{formatDateTime(item.job.created_at)}</td>
                <td>
                  <div className="dense-text">
                    {item.phase_display.label} ({item.phase_display.percent}%)
                  </div>
                </td>
                <td>
                  <button className="ghost-button small" onClick={(e) => {
                    e.stopPropagation();
                    onInspectJob(item.job.job_id);
                  }}>
                    检视详情
                  </button>
                </td>
              </tr>
            ))
          ) : (
            <tr>
              <td colSpan={7}>
                <div className="empty-state large">当前无任务。点击“新建任务”开始。</div>
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
