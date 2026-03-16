@echo off
setlocal

set "COLMAP=C:\Users\27252\Downloads\colmap-x64-windows-cuda\COLMAP.bat"
set "ROOT=E:\kykt\Coding\3.9\distortion_colmap"
set "IMAGES=%ROOT%\images"
set "SPARSE=%ROOT%\workspace\sparse\0"
set "DENSE=%ROOT%\workspace\dense"

if not exist "%SPARSE%\cameras.bin" (
  echo Sparse model not found: %SPARSE%
  echo Run run_colmap_auto.bat first.
  exit /b 1
)

call "%COLMAP%" image_undistorter ^
  --image_path "%IMAGES%" ^
  --input_path "%SPARSE%" ^
  --output_path "%DENSE%" ^
  --output_type COLMAP

call "%COLMAP%" patch_match_stereo ^
  --workspace_path "%DENSE%" ^
  --workspace_format COLMAP ^
  --PatchMatchStereo.geom_consistency true

call "%COLMAP%" stereo_fusion ^
  --workspace_path "%DENSE%" ^
  --workspace_format COLMAP ^
  --input_type geometric ^
  --output_path "%DENSE%\fused.ply"

echo.
echo Dense reconstruction finished.
echo Output: %DENSE%\fused.ply
endlocal
