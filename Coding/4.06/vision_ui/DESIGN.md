# DESIGN.md

## Product context

This project is a **local desktop client**, not a marketing website and not a mobile app.

It is a **3R / visual geometry model workbench** used to:
- create and dispatch model jobs
- inspect remote execution status
- compare MASt3R / MonST3R / Spann3R / Align3R / Fast3R / CUT3R
- review artifacts, logs, evaluation, deployment state, and AI-assisted summaries

The UI should feel like a **serious desktop workbench** for long sessions:
- stable
- dense
- structured
- status-first
- comparison-friendly
- comfortable for repeated use

This design system should optimize for:
- desktop productivity
- engineering clarity
- multi-panel workflows
- rapid status scanning
- long-running experiment management

Do **not** style this like a landing page, portfolio site, mobile app, or playful consumer dashboard.

---

## Chosen visual direction

Use a **Workbench Light** style inspired by:

- Linear
- Raycast
- Warp

Interpretation for this project:

- Linear = structure, hierarchy, calm order
- Raycast = polished desktop utility feel
- Warp = engineering workbench / runtime console energy

The result should be:

- light but not flat white
- compact but not cramped
- modern but not flashy
- premium but not decorative
- tool-like rather than page-like

---

## Core design principles

### 1. Tooling over marketing
Every screen should feel like part of a desktop tool.
Avoid hero-page aesthetics, oversized empty space, oversized illustrations, or decorative sections with no operational value.

### 2. Density with control
This app must surface many states at once.
Favor compact, readable layouts over oversized cards.
Information density is good when grouping and spacing remain disciplined.

### 3. Status-first hierarchy
Users should immediately recognize:
- what is running
- what is blocked
- what is finished
- what needs attention
- what is only planned vs actually runnable

Status should be visible before ornament.

### 4. Comparison-first workflows
The app is not only for single-task inspection.
It must support comparing:
- samples
- models
- deployment readiness
- evaluation results
- output completeness

Layouts should make side-by-side reading easy.

### 5. Long-session comfort
This is a desktop app used for extended periods.
Use low-fatigue contrast, restrained glow, minimal motion, and consistent spacing.
Light mode should feel calm, not washed out.

### 6. One system, not many mini-styles
All pages must feel like one product.
Overview, Create, Jobs, Sample Matrix, System, and Advisor should share the same design language.

---

## Visual language

### Overall tone

- light neutral base
- cool, technical, precise
- subtle premium feel
- no heavy glassmorphism
- no glossy marketing gradients
- no playful rounded toy aesthetic

### Surfaces
Use layered light surfaces with clear separation:

- app frame background
- panel background
- inset blocks
- selected state background
- hovered state background

The UI should rely more on:

- border definition
- tonal separation
- section grouping

than on blur or large shadows.

### Corners
Use **moderate** border radii.
Not tiny, not pill-heavy everywhere.

Suggested feel:
- panels: medium radius
- inputs/buttons: small-to-medium radius
- badges/pills: rounded, but restrained

### Shadows
Use subtle shadows only to clarify elevation.
Shadows should never dominate the look.
On desktop, borders and tonal contrast are more important than dramatic depth.

---

## Color system

### Base palette roles
Define colors by semantic role, not by brand decoration.

- `bg-app`: primary app background
- `bg-elevated`: raised panels
- `bg-subtle`: inner grouped areas
- `bg-hover`: hover state
- `bg-active`: selected state
- `line-default`: normal border/divider
- `line-strong`: stronger separator or focused card border
- `text-primary`: main text
- `text-secondary`: muted supporting text
- `text-faint`: disabled/low-priority text

### Accent roles
Use accents sparingly.

- `accent-primary`: current focus, active navigation, major CTA
- `accent-running`: running state / in-progress computation
- `accent-success`: completed / healthy / available
- `accent-warning`: caution / partial / stale / planned-with-risk
- `accent-danger`: failed / blocked / hard error
- `accent-info`: helpful but non-blocking contextual info

### Color behavior
- Active should usually use a cool blue or blue-violet
- Running can use blue-cyan
- Success should be green
- Warning should be amber
- Danger should be red

Do not overload the interface with many saturated colors at once.
Most of the UI should remain neutral.

---

## Typography

### Tone
Typography should feel technical and calm.
Not editorial. Not playful. Not luxury fashion.

### Hierarchy
Use a restrained type scale.
Avoid giant display headings.

Recommended hierarchy:
- App title / key page title: strong but compact
- Section headings: clear and functional
- Card headings: medium weight
- Meta labels: small uppercase or compact label style
- Supporting text: slightly muted
- Monospace: for job ids, paths, command-like data, metrics where appropriate

### Behavior
- Tighten headings slightly
- Keep body line-height comfortable
- Keep labels readable but compact

---

## Layout rules

### App shell
The app should feel like a desktop shell with three main structural zones:
1. global app header
2. primary navigation
3. current workspace content

### Navigation
Prefer a desktop-workbench navigation feel.
Navigation should be stable and always recognizable.

If using side navigation, it should feel compact and dense.
If using top navigation, it should still feel like a tool, not like a website tab strip.

Navigation items should show:
- label
- count or state badge when useful
- active state clearly

### Page content
Pages should be built from aligned panels and split layouts, not free-floating cards.

Use consistent patterns like:
- left list + right detail
- top summary + lower work area
- matrix/table-style compare blocks
- operational panels grouped by purpose

### Grid behavior
Prefer structured grids with predictable alignment.
Avoid loose masonry-style layouts.

### Responsive behavior
This is desktop-first.
Optimize for wide windows before thinking about narrow widths.
When space shrinks, collapse intelligently, but do not make it feel like a mobile page.

---

## Component language

### Panels
Panels are the core container primitive.
Panels should feel like workstation modules, not soft marketing cards.

Each panel should have:
- clear title / eyebrow / purpose
- internal grouping if dense
- obvious boundary from neighboring panels

### Status cards
Status cards must be immediately scannable.
Use:
- concise title
- bold current state
- short support copy
- semantic accent color

### Buttons
Buttons should feel efficient and desktop-native.

Use three levels:
- primary action
- secondary action
- ghost / utility action

Avoid oversized rounded CTA buttons.
This is a tool, not a sales page.

### Inputs
Inputs should be clean, compact, and structured.
Focus state should be clear but restrained.

### Badges / pills
Use badges for:
- state
- model family
- source type
- counts
- warnings

Badges should not become decorative clutter.

### Tables / matrices
For sample comparison and deployment inspection, prefer matrix/table logic over large card lists.
Rows and columns should be easy to scan.

### Progress
Progress bars and stage indicators must feel operational.
Keep them compact and precise.
Do not make them overly playful or animated.

### Log views
Logs should feel like embedded console surfaces:
- mono font
- light inset surface
- high readability
- clear separation from decorative UI

### Artifact blocks
Artifacts should be grouped by semantic type:
- core 3D output
- camera/trajectory
- confidence/diagnostics
- frame previews
- other

The grouping UI should support quick scanning first, deep inspection second.

---

## Motion and interaction

Use minimal, fast motion.

Allowed motion:
- hover emphasis
- selected state transitions
- panel reveal
- modal fade/scale-in
- progress changes

Avoid:
- bouncy motion
- large animated gradients
- decorative movement
- delayed transitions that slow tool use

Everything should feel responsive and professional.

---

## Page-specific guidance

### 1. Overview / Workbench
This should feel like a command center.
It should prioritize:
- current focus task
- system health
- queue/running summary
- shortcut actions
- model roadmap
- sample/evaluation status

Avoid making this page feel like a landing page.
It should feel operational within 2 seconds.

### 2. Create Job
This is a configuration workspace.
It should feel like a controlled form-based tool.

Prioritize:
- model choice clarity
- input source clarity
- upload flow clarity
- advanced params tucked away but still structured
- clear recommended presets

### 3. Jobs
This is the core desktop work area.
Use a strong split-pane feel:
- left = job list / filters / queue
- right = selected job detail

The selected job detail should feel like an inspector console, not a blog article.

### 4. Sample Matrix
This is a comparison-first page.
It should look more like a matrix/workbench than a list of marketing cards.
Rows should anchor around `sample_id`.
Columns or grouped cells should anchor around models.

### 5. System / Deployment
This should feel like a compact operations dashboard.
Emphasize:
- readiness
- blockers
- env/file/dir status
- cache freshness
- last errors

### 6. Advisor
This should look like an auxiliary evaluation lane inside the tool, not a separate chatbot product.
Keep it aligned with the same panel system, density, and evidence-first reading order.

---

## Current refinement priorities (2026-04-25)

### Product surface
- Keep the app framed as Workbench Light: a local desktop workbench for 3R / visual-geometry model work.
- Treat the current navigation as stable: Overview / Create / Jobs / Sample Matrix / System / Advisor.
- Keep operational tone over visual novelty; every refinement should improve scan speed, comparison, or handoff.

### Create
- Keep Create as a launch console with visible service, model, input, and source readiness before submission.
- Keep the model picker catalog-driven and explicit about runnable models vs catalog-only research models.
- Preserve the input staging matrix with filename, type, size, and direct remove action.

### Jobs
- Keep Jobs as the primary split-pane workbench: filtered list and queue tooling on the left, inspector detail on the right.
- Preserve keyboard flow: `/` focuses search, `J` / `K` moves selection, and search-box `Up` / `Down` steps through filtered results.
- Keep logs keyword narrowing inline and preserve latest/suspicious line copy shortcuts regardless of filter.

### Sample Matrix
- Keep the matrix as a comparison surface, not a card feed.
- Preserve sort/filter controls, row selection, bulk job-id copy, unassigned-job pool, and locate-job handoff.
- Keep score strength, metric count, model status, and artifact hints compact inside matrix cells.

### System
- Keep deployment readiness as a model-target matrix with directory, environment, files, and checkpoints visible before narrative text.
- Make blocked-model state reusable across Create, Sample Matrix, and System instead of duplicating wording.

### Engineering
- Split `client/src/App.tsx` into workspace-sized components, data hooks, shared config, and pure helpers.
- Run full end-to-end Spann3R and Fast3R validation through the desktop client.
- Tighten job bundle export, report export, manual evaluation, and Advisor draft contracts without changing the core information architecture.

---

## Styling dos

- Do use compact panel systems
- Do use strong section hierarchy
- Do use semantic state colors carefully
- Do use neutral light surfaces
- Do use monospace for job ids, paths, and technical metadata
- Do favor scanning speed over visual spectacle
- Do make compare views feel structured and disciplined

## Styling don'ts

- Do not make this look like a landing page
- Do not use oversized glassmorphism everywhere
- Do not overuse giant gradients
- Do not make components too rounded or toy-like
- Do not rely on giant hero text
- Do not leave excessive empty whitespace
- Do not make the system page look like marketing cards
- Do not make the jobs page feel like stacked blog sections

---

## Implementation guidance for AI agents

When building or restyling pages in this project:
- think desktop client first
- think workbench first
- think dense operational UI first
- prefer structured alignment over decorative novelty
- preserve the existing information architecture when possible
- improve hierarchy, contrast, scanning, and consistency before adding new visual ideas

If unsure, choose the option that makes the app feel more like:
- an IDE
- a runtime control panel
- a model experiment workbench

and less like:
- a startup homepage
- a mobile dashboard
- a glossy design showcase
