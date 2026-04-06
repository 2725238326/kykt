@echo off
set ROOT=%~dp0
set PYTHON=%ROOT%\.venv\Scripts\python.exe

if not exist "%PYTHON%" (
  echo Virtual environment not found: %PYTHON%
  exit /b 1
)

cd /d "%ROOT%"
netstat -ano | findstr /R /C:"127.0.0.1:8000 .*LISTENING" >nul
if not errorlevel 1 (
  echo Port 8000 is already in use. Close the old KYKT Vision UI terminal first, then run this again.
  exit /b 1
)

"%PYTHON%" -m uvicorn app:app --host 127.0.0.1 --port 8000
