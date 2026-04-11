use serde::Serialize;
use std::{
    env,
    fs::{self, OpenOptions},
    io::Write,
    net::{SocketAddr, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::Mutex,
    thread,
    time::{Duration, Instant},
};
use tauri::Manager;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

const BACKEND_HOST: &str = "127.0.0.1";
const BACKEND_PORT: u16 = 8765;
const BACKEND_WAIT_SECS: u64 = 20;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;

struct BackendProcess(Mutex<Option<Child>>);

struct BackendStatusState(Mutex<BackendStatus>);

#[derive(Clone, Serialize)]
struct BackendStatus {
    running: bool,
    managed_by_tauri: bool,
    message: String,
    backend_root: Option<String>,
    log_path: Option<String>,
}

#[tauri::command]
fn app_ready_message() -> &'static str {
    "KYKT Vision Client desktop shell is ready."
}

#[tauri::command]
fn backend_status(state: tauri::State<'_, BackendStatusState>) -> BackendStatus {
    state
        .0
        .lock()
        .map(|status| status.clone())
        .unwrap_or_else(|_| BackendStatus {
            running: false,
            managed_by_tauri: false,
            message: "backend status lock is poisoned".to_string(),
            backend_root: None,
            log_path: None,
        })
}

pub fn run() {
    let context = tauri::generate_context!();
    let app = tauri::Builder::default()
        .manage(BackendProcess(Mutex::new(None)))
        .manage(BackendStatusState(Mutex::new(BackendStatus {
            running: false,
            managed_by_tauri: false,
            message: "backend has not been checked yet".to_string(),
            backend_root: None,
            log_path: None,
        })))
        .setup(|app| {
            set_backend_status(
                &app.handle(),
                BackendStatus {
                    running: false,
                    managed_by_tauri: false,
                    message: "Checking local backend availability...".to_string(),
                    backend_root: None,
                    log_path: None,
                },
            );

            let app_handle = app.handle().clone();
            thread::spawn(move || {
                set_backend_status(
                    &app_handle,
                    BackendStatus {
                        running: false,
                        managed_by_tauri: false,
                        message: "Starting or reusing the local FastAPI backend...".to_string(),
                        backend_root: None,
                        log_path: None,
                    },
                );
                let status = ensure_backend(&app_handle);
                set_backend_status(&app_handle, status);
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![app_ready_message, backend_status])
        .build(context)
        .expect("failed to build KYKT Vision Client");

    app.run(|app_handle, event| {
        if matches!(event, tauri::RunEvent::ExitRequested { .. }) {
            let state = app_handle.state::<BackendProcess>();
            stop_backend(&state);
        }
    });
}

fn set_backend_status(app: &tauri::AppHandle, status: BackendStatus) {
    if let Some(state) = app.try_state::<BackendStatusState>() {
        if let Ok(mut guard) = state.0.lock() {
            *guard = status;
        }
    }
}

fn ensure_backend(app: &tauri::AppHandle) -> BackendStatus {
    if backend_is_listening() {
        return BackendStatus {
            running: true,
            managed_by_tauri: false,
            message: "FastAPI backend is already listening on 127.0.0.1:8765.".to_string(),
            backend_root: None,
            log_path: None,
        };
    }

    let backend_root = match find_backend_root(app) {
        Ok(path) => path,
        Err(message) => {
            return BackendStatus {
                running: false,
                managed_by_tauri: false,
                message,
                backend_root: None,
                log_path: None,
            }
        }
    };

    match spawn_backend(&backend_root) {
        Ok((child, log_path)) => {
            if let Some(state) = app.try_state::<BackendProcess>() {
                if let Ok(mut guard) = state.0.lock() {
                    *guard = Some(child);
                }
            }

            if wait_for_backend() {
                BackendStatus {
                    running: true,
                    managed_by_tauri: true,
                    message: "FastAPI backend was started by the desktop client.".to_string(),
                    backend_root: Some(backend_root.display().to_string()),
                    log_path: Some(log_path.display().to_string()),
                }
            } else {
                BackendStatus {
                    running: false,
                    managed_by_tauri: true,
                    message: "Backend process started, but port 8765 did not become ready in time.".to_string(),
                    backend_root: Some(backend_root.display().to_string()),
                    log_path: Some(log_path.display().to_string()),
                }
            }
        }
        Err(message) => BackendStatus {
            running: false,
            managed_by_tauri: false,
            message,
            backend_root: Some(backend_root.display().to_string()),
            log_path: None,
        },
    }
}

fn backend_is_listening() -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], BACKEND_PORT));
    TcpStream::connect_timeout(&addr, Duration::from_millis(250)).is_ok()
}

fn wait_for_backend() -> bool {
    let deadline = Instant::now() + Duration::from_secs(BACKEND_WAIT_SECS);
    while Instant::now() < deadline {
        if backend_is_listening() {
            return true;
        }
        thread::sleep(Duration::from_millis(350));
    }
    false
}

fn find_backend_root(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    if let Ok(root) = env::var("KYKT_BACKEND_ROOT") {
        let path = PathBuf::from(root);
        if is_backend_root(&path) {
            return Ok(path);
        }
    }

    let manifest_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(client_dir) = manifest_root.parent() {
        if let Some(project_root) = client_dir.parent() {
            if is_backend_root(project_root) {
                return Ok(project_root.to_path_buf());
            }
        }
    }

    if let Ok(exe) = env::current_exe() {
        for ancestor in exe.ancestors() {
            if is_backend_root(ancestor) {
                return Ok(ancestor.to_path_buf());
            }
        }
    }

    if let Ok(resource_dir) = app.path().resource_dir() {
        for candidate in [resource_dir.join("backend"), resource_dir] {
            if is_backend_root(&candidate) {
                return Ok(candidate);
            }
        }
    }

    Err(
        "Could not locate the FastAPI backend. Set KYKT_BACKEND_ROOT to the vision_ui directory."
            .to_string(),
    )
}

fn is_backend_root(path: &Path) -> bool {
    path.join("app.py").exists()
        && path.join("job_store.py").exists()
        && path.join(".venv").join("Scripts").join("python.exe").exists()
}

fn spawn_backend(backend_root: &Path) -> Result<(Child, PathBuf), String> {
    let python = backend_root
        .join(".venv")
        .join("Scripts")
        .join("python.exe");
    if !python.exists() {
        return Err(format!("Python venv was not found: {}", python.display()));
    }

    let log_dir = backend_root.join("local_jobs").join("_desktop");
    fs::create_dir_all(&log_dir)
        .map_err(|err| format!("Failed to create backend log directory: {err}"))?;
    let log_path = log_dir.join("backend.log");
    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|err| format!("Failed to open backend log: {err}"))?;
    writeln!(log_file, "\n=== KYKT desktop backend start ===")
        .map_err(|err| format!("Failed to write backend log: {err}"))?;

    let stdout = Stdio::from(
        log_file
            .try_clone()
            .map_err(|err| format!("Failed to clone backend log handle: {err}"))?,
    );
    let stderr = Stdio::from(log_file);

    let mut command = Command::new(python);
    command
        .current_dir(backend_root)
        .env("PYTHONUTF8", "1")
        .args([
            "-m",
            "uvicorn",
            "app:app",
            "--host",
            BACKEND_HOST,
            "--port",
            &BACKEND_PORT.to_string(),
        ])
        .stdout(stdout)
        .stderr(stderr);

    #[cfg(windows)]
    command.creation_flags(CREATE_NO_WINDOW);

    command
        .spawn()
        .map(|child| (child, log_path))
        .map_err(|err| format!("Failed to start FastAPI backend: {err}"))
}

fn stop_backend(state: &BackendProcess) {
    if let Ok(mut guard) = state.0.lock() {
        if let Some(mut child) = guard.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
