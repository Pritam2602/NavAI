@echo off
setlocal
cd /d "%~dp0"

set PY=venv\Scripts\python.exe
if not exist "%PY%" set PY=.venv_ok\Scripts\python.exe
if not exist "%PY%" (
  echo Missing virtual environment. Run install_windows.bat first.
  exit /b 1
)

"%PY%" navai_system.py --websocket %*

