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

export type JobPayload = {
  job: JobRecord;
  phase_display: PhaseDisplay;
  outputs: Array<{
    relative_path: string;
    display_name: string;
    url: string;
    is_image: boolean;
    is_pointcloud: boolean;
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
  result_summary: {
    markdown?: string | null;
    text?: string | null;
  } | null;
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
};

export type JobsListPayload = {
  jobs: Array<{
    job: JobRecord;
    phase_display: PhaseDisplay;
  }>;
  summary: BootstrapPayload["summary"];
};
