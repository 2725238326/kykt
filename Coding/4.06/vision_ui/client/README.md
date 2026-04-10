# KYKT Vision Client

React + TypeScript rebuild for the local vision desktop-style client. The desktop shell is now scaffolded with Tauri 2.

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

- existing FastAPI backend on `127.0.0.1:8000`
- Vite dev server through Tauri
- Tauri 2 desktop shell

## Current scope

- dashboard
- local job creation
- recent jobs
- selected job detail
- polling against FastAPI JSON APIs
- Apple-inspired visual direction
- output cards with local open / view / download actions
- Tauri 2 shell scaffold under `src-tauri`

## Next

- replace current Jinja entry pages as the default launch target
- add MonST3R-specific creation flow
- decide how Python backend should be embedded or bundled for release builds
