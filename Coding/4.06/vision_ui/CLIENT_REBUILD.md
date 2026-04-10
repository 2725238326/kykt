# Client Rebuild Direction

Last updated: 2026-04-10

## Recommendation

The best long-term client stack for this project is:

- Frontend UI: React + TypeScript
- Desktop shell: Tauri 2
- System integration: Rust only for shell / file / process / OS bridge
- Existing backend: keep FastAPI + SSH/SCP during the migration

## Why this is better than Go or pure Rust for UI

Go and Rust are both excellent for system-facing tools, but not the fastest way
to build a polished Apple-like interface.

- Apple-style UI needs high iteration speed, layout freedom, and motion polish.
- React + TypeScript is much faster for this part.
- Tauri lets us keep a strong web UI while still shipping a real local desktop app.

## Visual direction

Reference inspiration:

- `VoltAgent/awesome-design-md`
- Apple-style rules: black / light-gray rhythm, single blue accent, pill buttons,
  restrained shadows, large SF-like typography, product-first layout

## Migration strategy

### Phase 1

Build a new React client against the current FastAPI JSON endpoints.

### Phase 2

Wrap the same UI with Tauri and move local file / process / SSH helpers behind a desktop-safe shell layer.

### Phase 3

Replace the current Jinja pages as the primary interface once the new client covers:

- job creation
- recent jobs
- job detail
- progress polling
- output preview / download
- MonST3R task flow

## Current skeleton

A new client skeleton now lives under:

- `E:\\kykt\\Coding\\4.06\\vision_ui\\client`
