# vision_ui

Local desktop workbench for 3R / visual-geometry model execution, inspection, comparison, and deployment tracking.

## Current architecture

```text
Tauri desktop shell
  -> React + TypeScript client
  -> local FastAPI backend on 127.0.0.1:8765
  -> system ssh/scp
  -> remote model runners on /hdd3/kykt26
```

The product is desktop-first. It is not a marketing site and not a thin web wrapper.

## Current product surface

- Overview command center for focus job, runtime health, and quick navigation
- Create workspace with catalog-driven model selection, input staging matrix, and family-based parameter routing
- Jobs split-pane workbench with filters, batch actions, keyboard navigation, inspector detail, logs, evaluation, and artifact access
- Sample Matrix compare workspace backed by `/api/samples`, with sorting, filtering, bulk ID operations, report export, and locate-job handoff
- System / deployment console backed by `/api/deployment/status`
- Advisor as an auxiliary draft/evaluation lane
- Workbench Light design system aligned with `DESIGN.md`

## Current model route

- Active 3R route: MASt3R, MonST3R, Spann3R, Align3R, Fast3R, CUT3R
- Currently creatable in the client: DUSt3R, MASt3R, MonST3R, Spann3R, Fast3R
- Catalog-visible but blocked: Align3R, CUT3R
- Deferred frontier research: Pi3X, ZipMap, LingBot-Map

## Run locally

Desktop entry:

```powershell
powershell -ExecutionPolicy Bypass -File E:\kykt\Coding\4.06\vision_ui\start_desktop_client.ps1
```

Backend only:

```bash
uvicorn app:app --reload
```

Frontend build:

```bash
cd E:/kykt/Coding/4.06/vision_ui/client && npm run build
```

## Near-term next work

- Run full end-to-end Spann3R and Fast3R jobs through the desktop client
- Make blocked-model deployment state more explicit and reusable
- Continue splitting `client/src/App.tsx` into workspace-sized components, hooks, and helper modules
- Add job bundle export and continue tightening report/evaluation contracts
