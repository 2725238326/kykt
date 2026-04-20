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
  artifact_groups?: Array<{
    key: string;
    label: string;
    count: number;
    description?: string;
  }> | null;
  primary_artifacts?: Array<{
    role: string;
    label?: string;
    name: string;
    relative_path: string;
    note?: string;
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

export type AdvisorStatus = {
  enabled: boolean;
  configured: boolean;
  base_url: string;
  model: string;
  has_api_key?: boolean;
  message: string;
};

export type AdvisorConfig = AdvisorStatus & {
  temperature: number;
  max_tokens: number;
  system_prompt: string;
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
  evaluation?: EvaluationPayload | null;
  advisor_report?: AdvisorReport | null;
};

export type EvaluationPayload = {
  job_id: string;
  rubric_version?: number;
  score_min?: number;
  score_max?: number;
  updated_at?: string | null;
  structure_completeness?: number | null;
  trajectory_stability?: number | null;
  noise?: number | null;
  dynamic_handling?: number | null;
  depth_continuity?: number | null;
  presentation_usability?: number | null;
  noise_control?: number | null;
  depth_consistency?: number | null;
  notes?: string;
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
    family?: string;
    runner_status?: string;
    research_priority?: number;
    active_track?: boolean;
  }>;
  model_catalog?: Array<{
    value: string;
    label: string;
    description: string;
    family: string;
    source_types: string[];
    runner_status: string;
    research_priority: number;
    active_track: boolean;
    runnable: boolean;
  }>;
  source_types: Array<{
    value: string;
    label: string;
  }>;
  advisor?: AdvisorStatus;
};

export type SampleManifestItem = {
  id: string;
  source_type: string;
  status: string;
  purpose: string;
  required_models?: string[];
  optional_models?: string[];
  target_file_count?: string;
  target_duration_seconds?: string;
  seed_job_id?: string;
  manual_criteria?: string[];
};

export type SamplesPayload = {
  manifest: {
    last_updated?: string | null;
    purpose?: string;
    active_models?: string[];
    deferred_models?: string[];
    samples?: SampleManifestItem[];
    scoring?: Record<string, string[]>;
  };
  summary: {
    sample_count: number;
    status_counts: Record<string, number>;
    source_counts: Record<string, number>;
    required_model_counts: Record<string, number>;
  };
  model_catalog: NonNullable<BootstrapPayload["model_catalog"]>;
};

export type DeploymentStatusPayload = {
  host?: string | null;
  root: string;
  fetched_at?: string;
  directories: Array<{
    name: string;
    path: string;
    state: string;
    exists: boolean;
    readme_setup: boolean;
    size_bytes?: number | null;
  }>;
  conda_envs: Array<{
    component: string;
    env: string;
    exists: boolean;
    path?: string | null;
  }>;
  known_files: Array<{
    component: string;
    kind: string;
    need: string;
    relative_path: string;
    path: string;
    exists: boolean;
    size_bytes?: number | null;
  }>;
  summary: {
    missing_directories: number;
    missing_conda_envs: number;
    missing_required_files: number;
    warnings: number;
    ok: boolean;
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
