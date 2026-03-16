# Distortion COLMAP Workspace

## Purpose

This workspace is a clean COLMAP project built from the images originally stored in:

`E:\kykt\Coding\3.9\畸变\Picture`

It keeps the original Chinese-path materials untouched and provides an ASCII-only path for COLMAP CLI and GUI.

## Structure

- `images/`
  Source images copied from the distortion folder.
- `calibration/K.txt`
  Existing intrinsic matrix exported from your calibration step.
- `workspace/`
  COLMAP database and reconstruction outputs.
- `scripts/run_colmap_auto.bat`
  Recommended path. Let COLMAP estimate the camera model from the images.
- `scripts/run_colmap_with_known_k.bat`
  Use your calibrated `K` as a fixed pinhole camera approximation.
- `scripts/open_colmap_gui.bat`
  Launch COLMAP GUI.
- `scripts/run_colmap_dense.bat`
  Optional dense reconstruction starting from the existing sparse model.

## Recommended workflow

### Option A: Recommended

Run:

`scripts/run_colmap_auto.bat`

Reason:

- Your images come from a phone camera.
- The current folder only stores `K.txt`, not a distortion coefficient file.
- Letting COLMAP optimize a `SIMPLE_RADIAL` single-camera model is usually more stable.

### Option B: Use the existing K matrix

Run:

`scripts/run_colmap_with_known_k.bat`

Reason:

- This injects your calibrated intrinsic matrix:
  - fx = 1235.1011
  - fy = 1243.21242
  - cx = 631.202298
  - cy = 783.622421

Limitation:

- This path assumes a fixed `PINHOLE` camera and does not include distortion coefficients.
- If the phone images still contain non-negligible lens distortion, reconstruction quality may degrade.

## How to open in COLMAP GUI

After reconstruction finishes:

1. Run `scripts/open_colmap_gui.bat`
2. In COLMAP GUI:
   - `File -> Import model`
   - choose:
     `E:\kykt\Coding\3.9\distortion_colmap\workspace\sparse\0`

If the model exists, you can inspect cameras and sparse points directly in COLMAP.

## Optional dense reconstruction

If you want a denser point cloud instead of only a sparse model, run:

`scripts/run_colmap_dense.bat`

This will produce:

- `workspace/dense/`
- `workspace/dense/fused.ply`
