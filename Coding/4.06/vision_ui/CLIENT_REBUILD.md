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

### Desktop shell

The Tauri 2 shell has now been scaffolded under:

- `E:\kykt\Coding\4.06\vision_ui\client\src-tauri`

Current assumption:

- FastAPI backend still runs separately on `127.0.0.1:8000`
- Tauri wraps the rebuilt client as the local desktop shell
- release bundling of the Python backend is still a later step

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

## Current launch paths

Web entry:

- `E:\kykt\Coding\4.06\vision_ui\start_client_rebuild.ps1`

Desktop entry:

- `E:\kykt\Coding\4.06\vision_ui\start_desktop_client.ps1`

## Remaining high-priority work

- MonST3R-specific task creation and result flow
- better output grouping and richer result summary cards
- embed or bundle the Python backend for release builds
- switch the project default entry away from the legacy Jinja pages
