# Figure plan and prompts

Status (2026-05-13): the survey is built in LaTeX; `main.tex` carries four TikZ figures and five booktabs tables. AI-generated raster figures are not used. Paper-figure crops exist in `figures/` as cached material but are not embedded in `main.tex` (see "Paper-figure cache" below).

## Figure policy

- Prefer TikZ concept figures embedded in `main.tex`. Keep them thin-line, white-background, restrained.
- Paper figures may only be embedded after the original paper's reuse/license terms have been independently confirmed; until then, do not insert `\includegraphics` of paper crops into the manuscript.
- If a raster AI figure is later considered, the exact prompt, model/tool, date, and output filename must be recorded here before use.
- Do not use decorative images. Every figure must explain a relation that the prose uses.

## Current figure inventory (matches `main.tex`)

| id | type | location | purpose |
|---|---|---|---|
| `fig:lineage` | TikZ taxonomy | end of §2 | DUSt3R-rooted six-branch lineage (matching/SfM, many-view/unified, dynamic/4D, streaming memory, test-time, Gaussian output) |
| `fig:paradigm` | TikZ two-row pipeline | §2 (after the three-consequences paragraph) | Traditional SfM/MVS pipeline vs 3R feed-forward pipeline |
| `fig:memory` | TikZ block diagram | §6 | Long-sequence memory primitives: recurrent state / spatial pointer / hybrid memory / cache-and-filter |
| `fig:application` | TikZ flow + evidence note | §9 | Application path from images to report, with evidence-logging branch |

## Current table inventory (matches `main.tex`)

| id | location | purpose |
|---|---|---|
| `tab:foundation` | §4 | Foundation + many-view models (DUSt3R, MASt3R, MASt3R-SfM, Fast3R, MV-DUSt3R+, VGGT, MapAnything/Pow3R) |
| `tab:dynamic` | §5 | Dynamic 3R mechanism comparison (Align3R, MonST3R, POMATO, D²USt3R, Easi3R, RayMap3R) |
| `tab:memory` | §6 | Long-sequence memory primitives by class |
| `tab:testtime` | §7 | Test-time mechanisms and prior-input entry points |
| `tab:application` | §9 | Application evidence matrix by output layer |

## Paper-figure cache

`figures/` contains re-cropped Fig.1 images from page 1 of four foundational papers. State as of 2026-05-13 re-crop pass:

| file | size on disk | visual-verification status | notes |
|---|---|---|---|
| `dust3r_fig1.png` | 1365 × 630 | ✓ verified clean | pointmap outputs + rendered point clouds; no caption tail; no title block |
| `vggt_fig1.png` | 1380 × 465 | ✓ verified clean | house/garden reconstruction + camera frusta + depth map; prior title-block issue resolved |
| `monst3r_fig1.png` | 1310 × 435 | ✓ verified clean | video strip + dynamic point cloud + output labels (Video Depth / Intrinsics / Dynamic & Static Mask) |
| `cut3r_fig1.png` | 1385 × 360 | ⚠ unverified | Read returned `(media removed — rejected by API)`; dimensions plausible but not visually confirmed |

Raw page rasters (`*_p1-01.png`, 1530 × 1980) are kept alongside for re-cropping. Generated via `pdftocairo -r 150 -singlefile -png` on the corresponding paper PDFs.

These crops are **not** referenced from `main.tex`. To embed any of them later:

1. Confirm reuse permission for the specific paper (arXiv preprint license is not uniform; check the paper's stated copyright and venue policy).
2. Re-verify the crop visually.
3. Add `\includegraphics[width=...]{figures/<name>.png}` inside a `figure` environment with an attribution caption such as: 「图 X. 摘自 \citet{KEY}，仅用于综述说明，版权归原作者。」.
4. Record the embedding decision in `notes/work_log.md`.

## Optional re-crop hints (only if/when embedding is approved)

Last-known coordinate guesses (Python PIL, applied to 1530×1980 raw page rasters):

```python
crops_v2 = {
    'dust3r':  (90, 465, 90+1350, 465+615),    # trim caption tail
    'vggt':    (200, 320, 200+1100, 320+750),  # shift down past title block
    'monst3r': (130, 415, 130+1230, 415+760),  # verify first
    'cut3r':   (60, 270, 60+1410, 270+650),    # verify first
}
```

These have not been re-applied yet; they are a starting point for the next attempt.

## Figure-sourcing checklist

- [ ] If embedding paper figures, confirm per-paper reuse license (DUSt3R, MASt3R, Fast3R, VGGT, CUT3R, Spann3R, MonST3R papers all have separate copyright statements).
- [ ] Re-crop and re-verify visually before inserting.
- [ ] Add a generated/derived asset to `figures/` with a source note.
- [ ] Captions for paper figures must attribute the source paper via `\citet{...}` and avoid any quality claim.
- [ ] Do not let figure captions assert performance or deployment claims that the prose does not support.
