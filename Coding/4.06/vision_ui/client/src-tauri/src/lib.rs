#[tauri::command]
fn app_ready_message() -> &'static str {
    "KYKT Vision Client desktop shell is ready."
}

pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![app_ready_message])
        .run(tauri::generate_context!())
        .expect("failed to run KYKT Vision Client");
}
