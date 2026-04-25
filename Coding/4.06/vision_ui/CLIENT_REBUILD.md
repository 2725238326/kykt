# Client Rebuild Direction

Last updated: 2026-04-25

## Recommendation

The long-term client stack for this project is:

- Frontend UI: React + TypeScript
- Desktop shell: Tauri 2
- System integration: Rust only for shell / file / process / OS bridge
- Existing backend: keep FastAPI + SSH/SCP during the migration

## Why this remains the right stack

Go and Rust are both excellent for system-facing tools, but they are not the fastest way to iterate on a dense desktop workbench UI.

- React + TypeScript gives the best iteration speed for high-density product UI.
- Tauri lets us ship a real local desktop app without giving up the web-style workflow.
- The current app is no longer a generic rebuild exercise; it is now a workbench product with concrete operator workflows.

## Visual direction

Reference inspiration:

- `VoltAgent/awesome-design-md`
- Workbench Light rules: neutral light surfaces, compact hierarchy, restrained blue accent, semantic state colors, dense desktop layout, comparison-first workflows

Primary design source of truth:

- `E:\kykt\Coding\4.06\vision_ui\DESIGN.md`

## Current implementation state

### React client

The rebuilt client lives under:

- `E:\kykt\Coding\4.06\vision_ui\client`

It currently provides:

- Overview command center for focus job, runtime health, and quick navigation
- Create workspace driven by `model_catalog`
- runnable-vs-catalog model distinction with explicit launch blockers
- family-based parameter routing for DUSt3R / MASt3R / MonST3R / Spann3R / Fast3R flows
- input staging matrix with filename / type / size / remove actions
- Jobs split-pane workbench with filters, lane cards, batch actions, selection navigator, and keyboard navigation
- inspector-style task detail with summary, outputs, logs, manual evaluation, advisor lane, and input previews
- Sample Matrix compare workspace with sorting, filtering, bulk ID operations, compare-report export, and locate-job handoff
- System deployment console backed by `/api/deployment/status`
- Advisor as an auxiliary draft/evaluation lane
- Workbench Light styling aligned with `DESIGN.md`

### Desktop shell

The Tauri 2 shell is scaffolded under:

- `E:\kykt\Coding\4.06\vision_ui\client\src-tauri`

Current behavior:

- Tauri checks `127.0.0.1:8765` on startup.
- If no backend is listening, Tauri starts the local FastAPI backend from the `vision_ui` project root.
- Backend logs from the desktop-supervised process are written to `local_jobs\_desktop\backend.log`.
- The React client reads the Tauri backend status command and shows whether the backend was reused or started by the desktop shell.
- Full portable bundling of Python/.venv is still a later step; the current release executable uses the existing project `.venv`.

## Migration strategy

### Phase 1

Keep expanding the rebuilt React client until it fully covers the operational workflow.

### Phase 2

Use Tauri as the standard local desktop entry and move only the native-only pieces into Rust when there is a clear need.

### Phase 3

Keep the React client as the default product surface and reduce the legacy Jinja pages to fallback/debug usage only.

## Current launch paths

Web entry:

- `E:\kykt\Coding\4.06\vision_ui\start_client_rebuild.ps1`

Desktop entry:

- `E:\kykt\Coding\4.06\vision_ui\start_desktop_client.ps1`

Release outputs:

- Easy release folder: `E:\kykt\release\kykt_vision_client`
- `E:\kykt\Coding\4.06\vision_ui\client\src-tauri\target\release\kykt_vision_client.exe`
- `E:\kykt\Coding\4.06\vision_ui\client\src-tauri\target\release\bundle\nsis\KYKT Vision Client_0.1.0_x64-setup.exe`
- `E:\kykt\Coding\4.06\vision_ui\client\src-tauri\target\release\bundle\msi\KYKT Vision Client_0.1.0_x64_en-US.msi`

## Remaining high-priority work

- full end-to-end Spann3R and Fast3R validation through the rebuilt client
- split `client/src/App.tsx` into smaller workspace containers and hooks
- add job bundle export and keep tightening report/evaluation contracts
- decide whether to fully bundle Python/.venv for portable release builds
