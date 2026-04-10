# KYKT Vision Client

React + TypeScript rebuild skeleton for the local vision desktop-style client.

## Run

Start the existing FastAPI backend first:

```powershell
powershell -ExecutionPolicy Bypass -File E:\kykt\Coding\4.06\vision_ui\start.ps1
```

Then start the new client:

```powershell
cd E:\kykt\Coding\4.06\vision_ui\client
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

## Current scope

- dashboard
- local job creation
- recent jobs
- selected job detail
- polling against FastAPI JSON APIs
- Apple-inspired visual direction

## Next

- replace current Jinja entry pages
- add richer output handling
- add MonST3R-specific creation flow
- wrap with Tauri
