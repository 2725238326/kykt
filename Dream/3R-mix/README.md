# 3R-mix review workspace

Status: initialized 2026-05-11.

This directory is for a Typst-based survey on recent 3R / feed-forward 3D
reconstruction models and their surrounding lines: pointmap-based
reconstruction, dense matching, streaming memory, dynamic reconstruction,
test-time verification, pose-free Gaussian outputs, and application-facing
result delivery.

## Directory layout

- `main.typ`: current Typst survey scaffold.
- `bib.yaml`: starter bibliography in Typst YAML format.
- `src/`: future chapter files if the survey is split out of `main.typ`.
- `papers/`: local PDFs to be downloaded or copied from existing assets.
- `figures/`: generated or extracted figures.
- `notes/`: reading notes, paper cards, and model comparison tables.
- `simple-typst-thesis/`: upstream template clone from `zagoli/simple-typst-thesis`.
- `GENERATION_PROMPT.md`: prompt for the agent that will generate the survey.
- `NEW_CHAT_HANDOFF.md`: short handoff text for a fresh conversation.

## Current writing standard

Use normal academic Chinese. Avoid marketing phrases, slogans, and obvious AI
prose. Do not claim performance or novelty beyond the evidence. The survey can
state that Dream accumulated a comparative perspective through implementation
and planning work, but Dream3R itself should be used as experience/context, not
as the centerpiece of the 3R literature survey.

## Immediate next tasks

1. Build a complete model inventory from Dream documents and the prior
   `ppt/3r_models_survey/research_notes.md` notes.
2. Download missing papers into `papers/`, prioritizing official arXiv or
   conference PDFs.
3. Convert the starter outline in `main.typ` into a full review.
4. Generate at least three figures:
   - lineage map from DUSt3R to related models,
   - capability taxonomy by reconstruction regime,
   - application pipeline from inputs to deployable 3D results.
5. Compile with Typst and check that bibliography, figures, and cross-references
   render correctly.
