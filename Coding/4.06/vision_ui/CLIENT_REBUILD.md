# Client Rebuild Direction

Last updated: 2026-04-10

## Recommendation

The long-term client stack for this project is:

- Frontend UI: React + TypeScript
- Desktop shell: Tauri 2
- System integration: Rust only for shell / file / process / OS bridge
- Existing backend: keep FastAPI + SSH/SCP during the migration

## Why this is better than Go or pure Rust for UI

Go and Rust are both excellent for system-facing tools, but they are not the fastest way
to build a polished Apple-like interface.

- Apple-style UI needs high iteration speed, layout freedom, and motion polish.
- React + TypeScript is much faster for this part.
- Tauri lets us keep a strong web UI while still shipping a real local desktop app.

## Visual direction

Reference inspiration:

- `VoltAgent/awesome-design-md`
- Apple-style rules: black / light-gray rhythm, single blue accent, pill buttons,
  restrained shadows, large SF-like typography, product-first layout

## Current implementation state

### React client

The rebuilt client now lives under:

- `E:\kykt\Coding\4.06\vision_ui\client`

It already has:

- task creation against FastAPI JSON APIs
- recent jobs list with polling
- selected task detail with progress timeline
- output cards with view / download / local open
- summary panel generated from returned result metadata
- Apple-inspired visual baseline
- DUSt3R parameter panel for image-set jobs
- MonST3R parameter panel for video or frame-sequence jobs
- output recognition for images, PLY point clouds, GLB scenes, video files, trajectory text, and NPY artifacts

### Desktop shell

The Tauri 2 shell has now been scaffolded under:

- `E:\kykt\Coding\4.06\vision_ui\client\src-tauri`

Current assumption:

- Tauri checks `127.0.0.1:8765` on startup.
- If no backend is listening, Tauri starts the local FastAPI backend from the `vision_ui` project root.
- Backend logs from the desktop-supervised process are written to `local_jobs\_desktop\backend.log`.
- The React client reads the Tauri backend status command and shows whether the backend was reused or started by the desktop shell.
- Full portable bundling of Python/.venv is still a later step; the current release executable uses the existing project `.venv`.

## Migration strategy

### Phase 1

Keep expanding the rebuilt React client until it fully covers the current workflow.

### Phase 2

Use Tauri as the standard local desktop entry and move only the native-only pieces into Rust when there is a clear need.

### Phase 3

Replace the current Jinja pages as the primary interface once the new client covers:

- job creation
- recent jobs
- task detail
- progress polling
- output preview / download / local open
- MonST3R task flow

### Phase 4

After the first MonST3R validation examples are stable, make the React client the
default product surface and keep the old Jinja pages only as a fallback/debug
interface.

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

- first end-to-end MonST3R validation run through the rebuilt client
- result grouping for MonST3R artifacts such as GLB scene, trajectory, depth, confidence, and dynamic masks
- better output grouping and richer result summary cards
- fully portable Python/.venv bundling for release builds
- switch the project default entry away from the legacy Jinja pages
