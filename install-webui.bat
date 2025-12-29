@echo off
setlocal enabledelayedexpansion

set "ROOT=%~dp0"
set "VENV=%ROOT%.venv"
set "PY_BIN=%VENV%\\Scripts\\python.exe"
set "TORCH_MODE=%CODEX_TORCH_MODE%"
if "%TORCH_MODE%"=="" set "TORCH_MODE=auto"

REM Prefer a specific interpreter if provided.
set "BOOTSTRAP_PY="
if defined PYTHON (
  set "BOOTSTRAP_PY=%PYTHON%"
) else (
  REM Prefer the Windows launcher with an explicit version if available.
  where py >nul 2>&1
  if %errorlevel%==0 (
    py -3.12 -c "import sys" >nul 2>&1
    if %errorlevel%==0 (
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

REM ---------------------------------------------------------------------------
REM Torch install (auto-detect)
REM CODEX_TORCH_MODE=auto|cpu|cuda|skip
REM ---------------------------------------------------------------------------
if /I "%TORCH_MODE%"=="skip" (
  echo [install] Skipping torch install (CODEX_TORCH_MODE=skip).
  goto :after_torch
)

"%PY_BIN%" -c "import torch; print(torch.__version__)" >nul 2>&1
if %errorlevel%==0 (
  echo [install] torch already installed; skipping.
  goto :after_torch
)

echo [install] torch not found; attempting auto-install...

set "TORCH_VARIANTS="

if /I "%TORCH_MODE%"=="cpu" (
  set "TORCH_VARIANTS=cpu"
) else if /I "%TORCH_MODE%"=="cuda" (
  set "TORCH_VARIANTS=cu126 cu124 cu121 cu118"
) else (
  where nvidia-smi >nul 2>&1
  if %errorlevel%==0 (
    echo [install] nvidia-smi detected:
    where nvidia-smi
    echo [install] nvidia-smi GPU query (name, driver, cuda):
    nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader 2>nul
    set "CUDA_LINE="
    for /f "delims=" %%L in ('nvidia-smi ^| findstr /C:"CUDA Version"') do set "CUDA_LINE=%%L"
    set "CUDA_VER="
    if not "!CUDA_LINE!"=="" (
      set "CUDA_VER_RAW=!CUDA_LINE:*CUDA Version:=!"
      for /f "tokens=1 delims= " %%A in ("!CUDA_VER_RAW!") do set "CUDA_VER=%%A"
    )
    if not "%CUDA_VER%"=="" (
      echo [install] Detected NVIDIA driver CUDA version: %CUDA_VER%
      for /f "tokens=1,2 delims=." %%A in ("%CUDA_VER%") do (
        set "CUDA_MAJ=%%A"
        set "CUDA_MIN=%%B"
      )
      if "%CUDA_MIN%"=="" set "CUDA_MIN=0"

      set "TORCH_VARIANTS=cpu"
      if %CUDA_MAJ% GEQ 12 (
        if %CUDA_MIN% GEQ 6 set "TORCH_VARIANTS=cu126 cu124 cu121"
        if %CUDA_MIN% GEQ 4 if "%TORCH_VARIANTS%"=="cpu" set "TORCH_VARIANTS=cu124 cu121"
        if %CUDA_MIN% GEQ 1 if "%TORCH_VARIANTS%"=="cpu" set "TORCH_VARIANTS=cu121"
        if "%TORCH_VARIANTS%"=="cpu" set "TORCH_VARIANTS=cu118"
      ) else if %CUDA_MAJ% EQU 11 (
        if %CUDA_MIN% GEQ 8 set "TORCH_VARIANTS=cu118"
      )
    ) else (
      set "TORCH_VARIANTS=cpu"
    )
  ) else (
    set "TORCH_VARIANTS=cpu"
  )
)

echo [install] Torch wheel candidates: %TORCH_VARIANTS%

set "TORCH_OK=0"
for %%V in (%TORCH_VARIANTS%) do (
  if "!TORCH_OK!"=="0" (
    echo [install] Installing torch/torchvision (%%V) ...
    "%PY_BIN%" -m pip install --index-url https://download.pytorch.org/whl/%%V --extra-index-url https://pypi.org/simple torch torchvision
    if !errorlevel! EQU 0 (
      set "TORCH_OK=1"
    )
  )
)

if "%TORCH_OK%"=="0" (
  echo Error: failed to install torch automatically.>&2
  echo Set CODEX_TORCH_MODE=cpu to force CPU wheels, or CODEX_TORCH_MODE=skip to skip.>&2
  exit /b 1
)

:after_torch
echo [install] torch:
"%PY_BIN%" -c "import torch; print('torch', torch.__version__); print('cuda available?', torch.cuda.is_available())" 2>nul

echo [install] Installing Python requirements ...
"%PY_BIN%" -m pip install -r "%ROOT%requirements.txt"

echo [install] pip check:
"%PY_BIN%" -m pip check

echo [install] Key package versions:
"%PY_BIN%" -c "import importlib.metadata as m; print('diffusers', m.version('diffusers')); print('transformers', m.version('transformers')); print('peft', m.version('peft')); print('accelerate', m.version('accelerate')); print('huggingface-hub', m.version('huggingface-hub')); print('tokenizers', m.version('tokenizers')); print('safetensors', m.version('safetensors')); print('fastapi', m.version('fastapi')); print('uvicorn', m.version('uvicorn')); print('pydantic', m.version('pydantic')); print('numpy', m.version('numpy')); print('pillow', m.version('pillow'))"

echo [install] Installing frontend dependencies (npm) ...
where node >nul 2>&1
if %errorlevel% neq 0 (
  echo Warning: missing node on PATH; skipping frontend install.>&2
  echo Install Node.js (>=18), then run: (cd apps\\interface ^&^& npm install).>&2
  exit /b 0
)
where npm >nul 2>&1
if %errorlevel% neq 0 (
  echo Warning: missing npm on PATH; skipping frontend install.>&2
  echo Install Node.js (>=18), then run: (cd apps\\interface ^&^& npm install).>&2
  exit /b 0
)

for /f "delims=" %%V in ('node -v 2^>nul') do echo [install] node: %%V
for /f "delims=" %%V in ('npm -v 2^>nul') do echo [install] npm: %%V

pushd "%ROOT%apps\\interface"
call npm install
set "NPM_EXIT=%errorlevel%"
popd
if %NPM_EXIT% neq 0 (
  echo Error: npm install failed with exit code %NPM_EXIT%.>&2
  exit /b %NPM_EXIT%
)

echo.
echo [install] Done.
echo Run:
echo   run-webui.bat
