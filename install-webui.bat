@echo off
setlocal

set "ROOT=%~dp0"
set "UV_VERSION=%CODEX_UV_VERSION%"
if "%UV_VERSION%"=="" set "UV_VERSION=0.9.17"
set "UV_DIR=%ROOT%.uv\\bin"
set "UV_BIN=%UV_DIR%\\uv.exe"

set "PYTHON_VERSION=%CODEX_PYTHON_VERSION%"
if "%PYTHON_VERSION%"=="" set "PYTHON_VERSION=3.12.10"

set "VENV=%ROOT%.venv"

set "TORCH_MODE=%CODEX_TORCH_MODE%"
if "%TORCH_MODE%"=="" set "TORCH_MODE=auto"
set "TORCH_BACKEND=%CODEX_TORCH_BACKEND%"

echo [install] Repo: %ROOT%
echo [install] uv: %UV_BIN% (version pin: %UV_VERSION%)
echo [install] Python: %PYTHON_VERSION% (managed by uv)
echo [install] Venv: %VENV% (created by uv; uses the managed Python)
echo [install] Torch mode: %TORCH_MODE% (CODEX_TORCH_MODE=auto^|cpu^|cuda^|skip)
if not "%TORCH_BACKEND%"=="" echo [install] Torch backend override: %TORCH_BACKEND% (CODEX_TORCH_BACKEND)

if not exist "%UV_BIN%" (
  if not exist "%UV_DIR%" mkdir "%UV_DIR%"
  echo [install] Installing uv %UV_VERSION% into %UV_DIR% ...
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$env:UV_NO_MODIFY_PATH='1'; $env:UV_UNMANAGED_INSTALL='%UV_DIR%'; irm 'https://astral.sh/uv/%UV_VERSION%/install.ps1' | iex"
  if errorlevel 1 (
    echo Error: failed to install uv.>&2
    exit /b 1
  )
)

if not exist "%UV_BIN%" (
  echo Error: uv install succeeded but '%UV_BIN%' is missing.>&2
  exit /b 1
)

set "UV_PYTHON_INSTALL_DIR=%ROOT%.uv\\python"
set "UV_PYTHON_INSTALL_BIN=0"
set "UV_PYTHON_INSTALL_REGISTRY=0"
set "UV_PYTHON_PREFERENCE=only-managed"
set "UV_PYTHON_DOWNLOADS=manual"
set "UV_PROJECT_ENVIRONMENT=%VENV%"

echo [install] Installing managed Python %PYTHON_VERSION% ...
"%UV_BIN%" python install "%PYTHON_VERSION%"
if errorlevel 1 (
  echo Error: failed to install Python %PYTHON_VERSION% via uv.>&2
  exit /b 1
)

set "TORCH_EXTRA="
if not "%TORCH_BACKEND%"=="" (
  set "TORCH_EXTRA=%TORCH_BACKEND%"
) else if /i "%TORCH_MODE%"=="skip" (
  set "TORCH_EXTRA="
) else if /i "%TORCH_MODE%"=="cpu" (
  set "TORCH_EXTRA=cpu"
) else if /i "%TORCH_MODE%"=="cuda" (
  set "TORCH_EXTRA=cu126"
) else (
  where nvidia-smi >nul 2>&1
  if errorlevel 1 (
    set "TORCH_EXTRA=cpu"
  ) else (
    set "TORCH_EXTRA=cu126"
  )
)

if "%TORCH_EXTRA%"=="" (
  echo [install] Warning: skipping torch/torchvision install (CODEX_TORCH_MODE=skip). The WebUI will not run without PyTorch. 1>&2
  echo [install] Syncing Python dependencies (locked) ...
  "%UV_BIN%" sync --locked
) else (
  echo [install] Syncing Python dependencies (locked) with torch extra: %TORCH_EXTRA% ...
  "%UV_BIN%" sync --locked --extra "%TORCH_EXTRA%"
)
if errorlevel 1 (
  echo Error: uv sync failed.>&2
  exit /b 1
)

echo [install] Installing frontend dependencies (npm) ...
where node >nul 2>&1
if errorlevel 1 (
  echo [install] Warning: missing 'node' on PATH; skipping frontend install. 1>&2
  echo [install] Install Node.js (^>=18^), then run: cd apps\\interface ^&^& npm install 1>&2
  goto :done
)
where npm >nul 2>&1
if errorlevel 1 (
  echo [install] Warning: missing 'npm' on PATH; skipping frontend install. 1>&2
  echo [install] Install Node.js (^>=18^), then run: cd apps\\interface ^&^& npm install 1>&2
  goto :done
)

echo [install] node: 
node -v
echo [install] npm:
npm -v
pushd "%ROOT%apps\\interface"
npm install
if errorlevel 1 (
  popd
  echo Error: npm install failed.>&2
  exit /b 1
)
popd

:done
echo.
echo [install] Done.
echo [install] Run: run-webui.bat
