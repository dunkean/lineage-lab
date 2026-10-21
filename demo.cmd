@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 exit /b 1
    ".venv\Scripts\python.exe" -m pip install -r requirements.lock
    if errorlevel 1 exit /b 1
    ".venv\Scripts\python.exe" -m pip install -e . --no-deps
    if errorlevel 1 exit /b 1
)
echo Cousinade: http://127.0.0.1:8772
".venv\Scripts\python.exe" -m cousinade studio --port 8772
