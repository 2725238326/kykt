# KYKT Vision Client

React + TypeScript rebuild for the local vision desktop-style client. The desktop shell uses Tauri 2 and can supervise the local FastAPI backend.

## Run

Web dev entry:

```powershell
powershell -ExecutionPolicy Bypass -File E:\kykt\Coding\4.06\vision_ui\start_client_rebuild.ps1
```

Desktop dev entry:

```powershell
powershell -ExecutionPolicy Bypass -File E:\kykt\Coding\4.06\vision_ui\start_desktop_client.ps1
```

The web entry opens Vite on:

```text
http://127.0.0.1:5173
```

The desktop entry starts:

- FastAPI backend on `127.0.0.1:8765`
- Vite dev server through Tauri
- Tauri 2 desktop shell

Release app / installers:

```text
E:\kykt\release\kykt_vision_client\kykt_vision_client.exe
E:\kykt\release\kykt_vision_client\KYKT Vision Client_0.1.0_x64-setup.exe
E:\kykt\release\kykt_vision_client\KYKT Vision Client_0.1.0_x64_en-US.msi
E:\kykt\Coding\4.06\vision_ui\client\src-tauri\target\release\kykt_vision_client.exe
E:\kykt\Coding\4.06\vision_ui\client\src-tauri\target\release\bundle\nsis\KYKT Vision Client_0.1.0_x64-setup.exe
E:\kykt\Coding\4.06\vision_ui\client\src-tauri\target\release\bundle\msi\KYKT Vision Client_0.1.0_x64_en-US.msi
```

The release executable checks `127.0.0.1:8765` on startup. If the backend is not already running, it locates the `vision_ui` project root and starts `.venv\Scripts\python.exe -m uvicorn app:app --port 8765` in the background. The backend log is written to:

```text
E:\kykt\Coding\4.06\vision_ui\local_jobs\_desktop\backend.log
```

If the project root moves, set `KYKT_BACKEND_ROOT` to the `vision_ui` directory before launching the app.

## Current scope

- dashboard
- local job creation
- recent jobs
- selected job detail
- polling against FastAPI JSON APIs
- Apple-inspired visual direction
- output cards with local open / view / download actions
- Tauri 2 shell scaffold under `src-tauri`
- Tauri-managed local backend process for the release executable
- desktop UI backend status chip showing whether FastAPI was reused or started by Tauri

## Next

- replace current Jinja entry pages as the default launch target
- first end-to-end MonST3R validation through the desktop client
- decide whether to fully bundle Python/.venv for portable release builds
