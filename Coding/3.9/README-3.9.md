# 3.9 Folder Notes

## Current contents

- `Multiview-Structure-From-Motion-main/`
  Reference incremental SfM project using Python and a provided `K.txt`.
- `畸变/`
  Your own calibration and image folder.
- `distortion_colmap/`
  Clean COLMAP workspace created from `畸变/Picture`.

## Recommended usage split

### Reference code study

Use:

`E:\kykt\Coding\3.9\Multiview-Structure-From-Motion-main\Multiview-Structure-From-Motion-main`

Purpose:

- Read how `recoverPose`, `solvePnPRansac`, and triangulation are chained in an incremental SfM pipeline.

### Your own calibration work

Use:

`E:\kykt\Coding\3.9\畸变`

Purpose:

- Store your chessboard calibration code and intermediate files.

### Your own COLMAP reconstruction

Use:

`E:\kykt\Coding\3.9\distortion_colmap`

Purpose:

- Run COLMAP reconstruction on your own image set without fighting Chinese paths or mixed project assets.
