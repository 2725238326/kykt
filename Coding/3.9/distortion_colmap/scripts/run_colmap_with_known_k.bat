@echo off
setlocal

set "COLMAP=C:\Users\27252\Downloads\colmap-x64-windows-cuda\COLMAP.bat"
set "ROOT=E:\kykt\Coding\3.9\distortion_colmap"
set "IMAGES=%ROOT%\images"
set "WORKSPACE=%ROOT%\workspace"
set "DB=%WORKSPACE%\database_known_k.db"
set "SPARSE=%WORKSPACE%\sparse_known_k"
set "CAMERA_PARAMS=1235.1011,1243.21242,631.202298,783.622421"

if exist "%DB%" del /f /q "%DB%"
if not exist "%SPARSE%" mkdir "%SPARSE%"

call "%COLMAP%" feature_extractor ^
  --database_path "%DB%" ^
  --image_path "%IMAGES%" ^
  --ImageReader.single_camera 1 ^
  --ImageReader.camera_model PINHOLE ^
  --ImageReader.camera_params "%CAMERA_PARAMS%" ^
  --FeatureExtraction.use_gpu 0

call "%COLMAP%" exhaustive_matcher ^
  --database_path "%DB%" ^
  --FeatureMatching.use_gpu 0

call "%COLMAP%" mapper ^
  --database_path "%DB%" ^
  --image_path "%IMAGES%" ^
  --output_path "%SPARSE%" ^
  --Mapper.ba_refine_focal_length 0 ^
  --Mapper.ba_refine_principal_point 0 ^
  --Mapper.ba_refine_extra_params 0

echo.
echo Reconstruction finished with fixed K.
echo Sparse models are under: %SPARSE%
endlocal
