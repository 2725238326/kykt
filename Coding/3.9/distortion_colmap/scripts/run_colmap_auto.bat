@echo off
setlocal

set "COLMAP=C:\Users\27252\Downloads\colmap-x64-windows-cuda\COLMAP.bat"
set "ROOT=E:\kykt\Coding\3.9\distortion_colmap"
set "IMAGES=%ROOT%\images"
set "WORKSPACE=%ROOT%\workspace"
set "DB=%WORKSPACE%\database.db"
set "SPARSE=%WORKSPACE%\sparse"

if exist "%DB%" del /f /q "%DB%"
if not exist "%SPARSE%" mkdir "%SPARSE%"

call "%COLMAP%" feature_extractor ^
  --database_path "%DB%" ^
  --image_path "%IMAGES%" ^
  --ImageReader.camera_model SIMPLE_RADIAL ^
  --FeatureExtraction.use_gpu 0

call "%COLMAP%" exhaustive_matcher ^
  --database_path "%DB%" ^
  --FeatureMatching.use_gpu 0

call "%COLMAP%" mapper ^
  --database_path "%DB%" ^
  --image_path "%IMAGES%" ^
  --output_path "%SPARSE%"

echo.
echo Reconstruction finished.
echo Sparse models are under: %SPARSE%
endlocal
