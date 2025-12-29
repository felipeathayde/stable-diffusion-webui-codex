@echo off
setlocal

set "ROOT=%~dp0"
set "VENV=%ROOT%.venv"
set "PY_BIN=%VENV%\Scripts\python.exe"
set "TORCH_MODE=%CODEX_TORCH_MODE%"
if "%TORCH_MODE%"=="" set "TORCH_MODE=auto"

REM Prefer a specific interpreter if provided.
set "BOOTSTRAP_PY="
if defined PYTHON (
  set "BOOTSTRAP_PY=%PYTHON%"
) else (
  REM Prefer the Windows launcher with an explicit version if available.
  where py >nul 2>&1
  if not errorlevel 1 (
    py -3.12 -c "import sys" >nul 2>&1
    if not errorlevel 1 (
      set "BOOTSTRAP_PY=py -3.12"
    ) else (
      set "BOOTSTRAP_PY=py"
    )
  ) else (
    set "BOOTSTRAP_PY=python"
  )
)

echo [install] Repo: %ROOT%
echo [install] Venv: %VENV%
echo [install] Torch mode: %TORCH_MODE% (CODEX_TORCH_MODE=auto^|cpu^|cuda^|skip)
echo [install] Bootstrap Python: %BOOTSTRAP_PY%

if not exist "%VENV%" (
  echo [install] Creating venv at %VENV% ...
  %BOOTSTRAP_PY% -m venv "%VENV%"
)

if not exist "%PY_BIN%" (
 echo Error: expected venv python at "%PY_BIN%".>&2
 exit /b 1
)

echo [install] Venv Python: %PY_BIN%
"%PY_BIN%" -c "import sys, platform; print('python', sys.version.split()[0]); print('exe', sys.executable); print('platform', platform.platform())"
"%PY_BIN%" -m pip --version

echo [install] Upgrading pip tooling ...
"%PY_BIN%" -m pip install -U pip wheel setuptools
"%PY_BIN%" -m pip --version

echo [install] Running Python installer (verbose) ...
"%PY_BIN%" "%ROOT%tools\\install_webui.py"
if errorlevel 1 (
  echo Error: installer failed.>&2
  exit /b 1
)
