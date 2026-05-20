@echo off
setlocal
cd /d "%~dp0"

set PY=venv\Scripts\python.exe
if not exist "%PY%" set PY=.venv_ok\Scripts\python.exe
if not exist "%PY%" (
  python -m venv venv
  set PY=venv\Scripts\python.exe
)

"%PY%" -m pip install --upgrade pip
echo.
echo If PyTorch is not installed yet, run this manually first:
echo "%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
echo.
"%PY%" -m pip install -r requirements-app.txt
"%PY%" tools\check_env.py

