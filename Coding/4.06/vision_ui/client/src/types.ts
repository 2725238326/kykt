export type JobRecord = {
  job_id: string;
  created_at: string;
  model: string;
  source_type: string;
  notes: string;
  params: Record<string, unknown>;
  status: string;
  phase: string;
  input_files: string[];
  input_items: Array<{
    original_name: string;
    stored_name: string;
    relative_path: string;
    size_bytes: number | null;
  }>;
  output_files: string[];
  remote_job_dir?: string | null;
  remote_runner?: string | null;
  error_message?: string | null;
  progress_message?: string | null;
};

export type PhaseDisplay = {
  label: string;
  description: string;
  percent: number;
  steps: Array<{
    code: string;
    label: string;
    hint: string;
    state: "todo" | "current" | "done";
  }>;
};

export type ResultSummary = {
  job_id: string;
  model: string;
  status: string;
  status_label: string;
  source_type: string;
  created_at: string;
  generated_at: string;
  duration_seconds?: number | null;
  inputs?: {
    count?: number;
    names?: string[];
  } | null;
  artifacts?: Array<{
    name: string;
    relative_path: string;
  }> | null;
  params?: Record<string, unknown> | null;
  scene_meta?: Record<string, unknown> | null;
  highlights?: string[] | null;
  next_actions?: string[] | null;
};

export type AdvisorReport = {
  evaluated_at: string;
  job_id: string;
  overall_score: number;
  readiness: string;
  summary: string;
  issues: string[];
  next_actions: string[];
  teacher_talk: string;
  advisor_model: string;
};

export type JobPayload = {
  job: JobRecord;
  phase_display: PhaseDisplay;
  outputs: Array<{
    relative_path: string;
    display_name: string;
    url: string;
    is_image: boolean;
    is_pointcloud: boolean;
    is_model3d: boolean;
    is_video: boolean;
    is_log: boolean;
  }>;
  previews: Array<{
    relative_path: string;
    display_name: string;
    stored_name: string;
    url: string;
    is_image: boolean;
  }>;
  logs: Array<{
    name: string;
    relative_path: string;
    tail: string;
  }>;
  result_summary: ResultSummary | null;
  advisor_report?: AdvisorReport | null;
};

export type BootstrapPayload = {
  summary: {
    total: number;
    running: number;
    finished: number;
    failed: number;
    cancelled: number;
  };
  delivery_gaps: Array<{
    title: string;
    detail: string;
  }>;
  server: {
    alias: string;
    host: string;
    user: string;
    port: number;
    remote_root: string;
  };
  models: Array<{
    value: string;
    label: string;
    description: string;
  }>;
  source_types: Array<{
    value: string;
    label: string;
  }>;
  advisor?: {
    enabled: boolean;
    configured: boolean;
    base_url: string;
    model: string;
    message: string;
  };
};

export type JobsListPayload = {
  jobs: Array<{
    job: JobRecord;
    phase_display: PhaseDisplay;
  }>;
  summary: BootstrapPayload["summary"];
};

export type BackendStatusPayload = {
  running: boolean;
  managed_by_tauri: boolean;
  message: string;
  backend_root: string | null;
  log_path: string | null;
};
