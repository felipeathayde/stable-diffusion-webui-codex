@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "ROOT=%~dp0"

REM --------------------------------------------------
REM Interactive menu (Windows)
REM --------------------------------------------------
REM By default, show a Simple vs Advanced menu so users don't need to set env vars manually.
REM For automation / CI, pass --no-menu (or pre-set CODEX_* env vars).

:restart
set "CODEX_MENU_USED="
set "CODEX_MENU_CANCEL="
set "CODEX_MENU_RERUN="

set "SHOW_MENU=1"
set "FORCE_MENU="

REM Args:
REM   --no-menu   Skip prompts
REM   --menu      Force prompts even if env vars are set
REM   --simple    Equivalent to AUTO (no prompts)
if /i "%~1"=="--no-menu" set "SHOW_MENU="
if /i "%~1"=="--simple" (
  set "SHOW_MENU="
  set "CODEX_TORCH_MODE=auto"
  set "CODEX_TORCH_BACKEND="
  set "CODEX_CUDA_VARIANT="
)
if /i "%~1"=="--menu" (
  set "SHOW_MENU=1"
  set "FORCE_MENU=1"
)

if defined CODEX_NO_MENU set "SHOW_MENU="
if defined CI set "SHOW_MENU="
if defined GITHUB_ACTIONS set "SHOW_MENU="

if not defined FORCE_MENU (
  if defined CODEX_TORCH_MODE set "SHOW_MENU="
  if defined CODEX_TORCH_BACKEND set "SHOW_MENU="
  if defined CODEX_CUDA_VARIANT set "SHOW_MENU="
  if defined CODEX_PYTHON_VERSION set "SHOW_MENU="
  if defined CODEX_UV_VERSION set "SHOW_MENU="
)
set "FORCE_MENU="

if defined SHOW_MENU (
  call :ui_menu
  if errorlevel 1 exit /b 1
  if defined CODEX_MENU_CANCEL exit /b 0
)
set "SHOW_MENU="

set "UV_VERSION=%CODEX_UV_VERSION%"
if "%UV_VERSION%"=="" set "UV_VERSION=0.9.17"
set "UV_DIR=%ROOT%.uv\bin"
set "UV_BIN=%UV_DIR%\uv.exe"

set "PYTHON_VERSION=%CODEX_PYTHON_VERSION%"
if "%PYTHON_VERSION%"=="" set "PYTHON_VERSION=3.12.10"

set "VENV=%ROOT%.venv"

set "TORCH_MODE=%CODEX_TORCH_MODE%"
if "%TORCH_MODE%"=="" set "TORCH_MODE=auto"
set "TORCH_BACKEND=%CODEX_TORCH_BACKEND%"
set "CUDA_VARIANT=%CODEX_CUDA_VARIANT%"

echo [install] Repo: %ROOT%
echo [install] uv: %UV_BIN%  version pin: %UV_VERSION%
echo [install] Python: %PYTHON_VERSION%  managed by uv
echo [install] Venv: %VENV%  created by uv; uses the managed Python
echo [install] Torch mode: %TORCH_MODE%  CODEX_TORCH_MODE=auto^|cpu^|cuda^|rocm^|skip
if not "%TORCH_BACKEND%"=="" echo [install] Torch backend override: %TORCH_BACKEND%  CODEX_TORCH_BACKEND
if not "%CUDA_VARIANT%"=="" echo [install] CUDA variant override: %CUDA_VARIANT%  CODEX_CUDA_VARIANT

if exist "%UV_BIN%" goto :uv_ok
if not exist "%UV_DIR%" mkdir "%UV_DIR%"
echo [install] Installing uv %UV_VERSION% into %UV_DIR% ...
call :install_uv
if not errorlevel 1 goto :uv_ok
echo Error: failed to install uv.>&2
exit /b 1
:uv_ok

if not exist "%UV_BIN%" (
  echo Error: uv install succeeded but '%UV_BIN%' is missing.>&2
  exit /b 1
)

set "UV_PYTHON_INSTALL_DIR=%ROOT%.uv\python"
set "UV_PYTHON_INSTALL_BIN=0"
set "UV_PYTHON_INSTALL_REGISTRY=0"
set "UV_PYTHON_PREFERENCE=only-managed"
set "UV_PYTHON_DOWNLOADS=manual"
set "UV_PROJECT_ENVIRONMENT=%VENV%"

echo [install] Installing managed Python %PYTHON_VERSION% ...
"%UV_BIN%" python install "%PYTHON_VERSION%"
if not errorlevel 1 goto :python_ok
echo Error: failed to install Python %PYTHON_VERSION% via uv.>&2
exit /b 1
:python_ok

set "TORCH_EXTRA="
if "%TORCH_BACKEND%"=="" goto :torch_mode_start
set "TORCH_EXTRA=%TORCH_BACKEND%"
goto :torch_extra_done

:torch_mode_start
if /i "%TORCH_MODE%"=="skip" goto :torch_extra_done
if /i "%TORCH_MODE%"=="cpu" goto :torch_mode_cpu
if /i "%TORCH_MODE%"=="rocm" goto :torch_mode_rocm
if /i "%TORCH_MODE%"=="cuda" goto :torch_mode_cuda
goto :torch_mode_auto

:torch_mode_cpu
set "TORCH_EXTRA=cpu"
goto :torch_extra_done

:torch_mode_rocm
echo [install] Warning: ROCm is Linux-only; falling back to CPU. 1>&2
set "TORCH_EXTRA=cpu"
goto :torch_extra_done

:torch_mode_cuda
call :pick_cuda_variant
goto :torch_extra_done

:torch_mode_auto
where nvidia-smi >nul 2>&1
if not errorlevel 1 goto :torch_auto_have_nvidia
set "TORCH_EXTRA=cpu"
goto :torch_extra_done

:torch_auto_have_nvidia
if "%CUDA_VARIANT%"=="" goto :torch_auto_detect_cuda
call :pick_cuda_variant
goto :torch_extra_done

:torch_auto_detect_cuda

set "CUDA_VER="
for /f "delims=" %%i in ('nvidia-smi --query-gpu=cuda_version --format=csv,noheader 2^>nul') do if not defined CUDA_VER set "CUDA_VER=%%i"
set "DRIVER_VER="
for /f "delims=" %%i in ('nvidia-smi --query-gpu=driver_version --format=csv,noheader 2^>nul') do if not defined DRIVER_VER set "DRIVER_VER=%%i"
set "GPU_NAME="
for /f "delims=" %%i in ('nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul') do if not defined GPU_NAME set "GPU_NAME=%%i"

set "DRIVER_MAJOR="
for /f "tokens=1 delims=." %%a in ("%DRIVER_VER%") do set "DRIVER_MAJOR=%%a"
set "CUDA_MAJOR="
set "CUDA_MINOR=0"
for /f "tokens=1 delims=." %%a in ("%CUDA_VER%") do set "CUDA_MAJOR=%%a"
for /f "tokens=2 delims=." %%a in ("%CUDA_VER%") do set "CUDA_MINOR=%%a"
if "%CUDA_MINOR%"=="" set "CUDA_MINOR=0"

REM Defaults / heuristics
set "TORCH_EXTRA=cu128"

REM NOTE: cmd.exe pre-parses multi-line (...) blocks. If a line expands to
REM "if  LSS 525 (...)" (missing left operand), cmd can crash with:
REM "525 was unexpected at this time." even if the outer IF condition would be false.
REM Keep numeric comparisons off nested blocks and guard empties before comparing.

set "DRIVER_MAJOR_NUM="
if "%DRIVER_MAJOR%"=="" goto :driver_major_num_done
set /a DRIVER_MAJOR_NUM=%DRIVER_MAJOR% 2>nul
if errorlevel 1 set "DRIVER_MAJOR_NUM="
:driver_major_num_done

REM Old NVIDIA drivers cannot run modern CUDA wheels; fall back to CPU (< 525).
if "%DRIVER_MAJOR_NUM%"=="" goto :torch_auto_cuda13
if %DRIVER_MAJOR_NUM% LSS 525 goto :torch_driver_too_old

REM CUDA 13 wheels require very new drivers; otherwise use CUDA 12.8 wheels.
:torch_auto_cuda13
if not "%CUDA_MAJOR%"=="13" goto :torch_auto_gpu_name
set "TORCH_EXTRA=cu128"
if "%DRIVER_MAJOR_NUM%"=="" goto :torch_extra_done
if %DRIVER_MAJOR_NUM% GEQ 580 set "TORCH_EXTRA=cu130"
goto :torch_extra_done

:torch_auto_gpu_name
echo %GPU_NAME% | findstr /i /r "RTX[ ]*50 RTX[ ]*5[0-9][0-9][0-9]" >nul 2>&1
if errorlevel 1 goto :torch_auto_cuda12
set "TORCH_EXTRA=cu128"
goto :torch_extra_done

:torch_auto_cuda12

if not "%CUDA_MAJOR%"=="12" goto :torch_extra_done
set /a __tmp=%CUDA_MINOR% 2>nul
if errorlevel 1 goto :torch_extra_done
if %CUDA_MINOR% GEQ 8 goto :torch_cuda12_cu128
if %CUDA_MINOR% GEQ 6 goto :torch_cuda12_cu126
set "TORCH_EXTRA=cu126"
goto :torch_extra_done

:torch_cuda12_cu128
set "TORCH_EXTRA=cu128"
goto :torch_extra_done

:torch_cuda12_cu126
set "TORCH_EXTRA=cu126"
goto :torch_extra_done

:torch_driver_too_old
set "TORCH_EXTRA=cpu"
goto :torch_extra_done

:torch_extra_done

if "%TORCH_EXTRA%"=="" goto :uv_sync_no_torch

echo [install] Syncing Python dependencies [locked] with torch extra: %TORCH_EXTRA% ...
"%UV_BIN%" sync --locked --extra "%TORCH_EXTRA%"
goto :uv_sync_done

:uv_sync_no_torch
echo [install] Warning: skipping torch/torchvision install. CODEX_TORCH_MODE=skip. WebUI requires PyTorch. 1>&2
echo [install] Syncing Python dependencies [locked] ...
"%UV_BIN%" sync --locked

:uv_sync_done
if not errorlevel 1 goto :uv_sync_ok
echo Error: uv sync failed.>&2
exit /b 1
:uv_sync_ok

echo [install] Installing frontend dependencies ...
where node >nul 2>&1
if errorlevel 1 goto :frontend_missing_node
where npm >nul 2>&1
if errorlevel 1 goto :frontend_missing_npm

echo [install] node: 
node -v
echo [install] npm:
npm -v
pushd "%ROOT%apps\\interface"
npm install
if not errorlevel 1 goto :frontend_install_ok
popd
echo Error: npm install failed.>&2
exit /b 1

:frontend_install_ok
popd
goto :done

:frontend_missing_node
echo [install] Warning: missing 'node' on PATH; skipping frontend install. 1>&2
echo [install] Install Node.js 18+ then execute:
echo [install]   cd apps\\interface
echo [install]   npm install
goto :done

:frontend_missing_npm
echo [install] Warning: missing 'npm' on PATH; skipping frontend install. 1>&2
echo [install] Install Node.js 18+ then execute:
echo [install]   cd apps\\interface
echo [install]   npm install
goto :done

:done
echo.
echo [install] Done.
echo [install] Next: run-webui.bat

if defined CODEX_MENU_USED (
  call :ui_menu_post
  if defined CODEX_MENU_RERUN goto :restart
)

exit /b 0

:ui_menu
set "CODEX_MENU_USED=1"

:ui_menu_loop
cls
echo ==================================================
echo  Codex WebUI Installer (Windows)
echo ==================================================
echo.
echo  [1] Simple install (AUTO)
echo  [2] Advanced (choose backend / CUDA)
echo  [3] Exit
echo.
set "CHOICE="
set /p "CHOICE=Select an option (1-3): "

if "%CHOICE%"=="1" goto :ui_menu_set_simple
if "%CHOICE%"=="2" goto :ui_menu_advanced
if "%CHOICE%"=="3" goto :ui_menu_cancel
goto :ui_menu_loop

:ui_menu_set_simple
set "CODEX_TORCH_MODE=auto"
set "CODEX_TORCH_BACKEND="
set "CODEX_CUDA_VARIANT="
exit /b 0

:ui_menu_cancel
set "CODEX_MENU_CANCEL=1"
exit /b 0

:ui_menu_advanced
cls
echo ==================================================
echo  Advanced Install Options
echo ==================================================
echo.
echo  Choose a backend:
echo   [1] AUTO (default)
echo   [2] CPU
echo   [3] CUDA (pick 12.6/12.8/13)
echo   [4] SKIP torch install (not recommended)
echo   [5] Back
echo.
set "BCHOICE="
set /p "BCHOICE=Select backend (1-5): "

if "%BCHOICE%"=="5" goto :ui_menu_loop

REM Reset only what this UI owns.
set "CODEX_TORCH_MODE="
set "CODEX_TORCH_BACKEND="
set "CODEX_CUDA_VARIANT="

if "%BCHOICE%"=="1" (
  set "CODEX_TORCH_MODE=auto"
  exit /b 0
)
if "%BCHOICE%"=="2" (
  set "CODEX_TORCH_MODE=cpu"
  exit /b 0
)
if "%BCHOICE%"=="3" goto :ui_menu_cuda
if "%BCHOICE%"=="4" (
  set "CODEX_TORCH_MODE=skip"
  exit /b 0
)
goto :ui_menu_advanced

:ui_menu_cuda
cls
echo ==================================================
echo  CUDA Variant
echo ==================================================
echo.
echo  Pick a CUDA wheel family (PyTorch):
echo   [1] CUDA 12.6  (cu126)
echo   [2] CUDA 12.8  (cu128)  [recommended]
echo   [3] CUDA 13    (cu130)
echo   [4] Back
echo.
set "CCHOICE="
set /p "CCHOICE=Select CUDA (1-4): "

if "%CCHOICE%"=="4" goto :ui_menu_advanced

set "CODEX_TORCH_MODE=cuda"
if "%CCHOICE%"=="1" set "CODEX_CUDA_VARIANT=12.6"
if "%CCHOICE%"=="2" set "CODEX_CUDA_VARIANT=12.8"
if "%CCHOICE%"=="3" set "CODEX_CUDA_VARIANT=13"

if "%CODEX_CUDA_VARIANT%"=="" goto :ui_menu_cuda
exit /b 0

:ui_menu_post
echo.
echo ==================================================
echo  Installer finished.
echo ==================================================
echo.
echo  [1] Back to menu
echo  [2] Exit
echo.
set "PCHOICE="
set /p "PCHOICE=Select an option (1-2): "
if "%PCHOICE%"=="1" set "CODEX_MENU_RERUN=1"
exit /b 0

:pick_cuda_variant
if /i "%CUDA_VARIANT%"=="12.6" goto :pick_cuda_cu126
if /i "%CUDA_VARIANT%"=="cu126" goto :pick_cuda_cu126
if /i "%CUDA_VARIANT%"=="12.8" goto :pick_cuda_cu128
if /i "%CUDA_VARIANT%"=="cu128" goto :pick_cuda_cu128
if /i "%CUDA_VARIANT%"=="13" goto :pick_cuda_cu130
if /i "%CUDA_VARIANT%"=="cu130" goto :pick_cuda_cu130
set "TORCH_EXTRA=cu128"
exit /b 0

:pick_cuda_cu126
set "TORCH_EXTRA=cu126"
exit /b 0

:pick_cuda_cu128
set "TORCH_EXTRA=cu128"
exit /b 0

:pick_cuda_cu130
set "TORCH_EXTRA=cu130"
exit /b 0

:install_uv
setlocal EnableExtensions

REM Prefer downloading the prebuilt zip from GitHub Releases to avoid PowerShell module issues
REM (some environments cannot load Microsoft.PowerShell.Security, breaking the installer script).

set "ARCH=%PROCESSOR_ARCHITECTURE%"
if defined PROCESSOR_ARCHITEW6432 set "ARCH=%PROCESSOR_ARCHITEW6432%"
set "UV_ASSET="
if /i "%ARCH%"=="AMD64" set "UV_ASSET=uv-x86_64-pc-windows-msvc.zip"
if /i "%ARCH%"=="ARM64" set "UV_ASSET=uv-aarch64-pc-windows-msvc.zip"
if /i "%ARCH%"=="x86" set "UV_ASSET=uv-i686-pc-windows-msvc.zip"

if "%UV_ASSET%"=="" goto :install_uv_unsupported_arch

set "UV_URL=https://github.com/astral-sh/uv/releases/download/%UV_VERSION%/%UV_ASSET%"
set "UV_ZIP=%UV_DIR%\%UV_ASSET%"

if exist "%UV_ZIP%" del /f /q "%UV_ZIP%" >nul 2>&1

where curl >nul 2>&1
if not errorlevel 1 goto :install_uv_download_curl

where certutil >nul 2>&1
if not errorlevel 1 goto :install_uv_download_certutil

echo Error: neither curl nor certutil is available to download uv.>&2
endlocal
exit /b 1

:install_uv_download_curl
echo [install] Downloading: %UV_URL%
curl -L --fail --retry 3 --retry-delay 2 -o "%UV_ZIP%" "%UV_URL%"
if errorlevel 1 goto :install_uv_download_failed
goto :install_uv_extract

:install_uv_download_certutil
echo [install] Downloading via certutil: %UV_URL%
certutil -urlcache -split -f "%UV_URL%" "%UV_ZIP%" >nul
if errorlevel 1 goto :install_uv_download_failed

:install_uv_extract
where tar >nul 2>&1
if not errorlevel 1 goto :install_uv_extract_tar
goto :install_uv_extract_ps

:install_uv_extract_tar
echo [install] Extracting uv via tar ...
tar -xf "%UV_ZIP%" -C "%UV_DIR%"
if errorlevel 1 goto :install_uv_extract_failed
goto :install_uv_post_extract

:install_uv_extract_ps
echo [install] Extracting uv via PowerShell ZipFile ...
powershell -NoProfile -Command "Add-Type -AssemblyName System.IO.Compression.FileSystem; [System.IO.Compression.ZipFile]::ExtractToDirectory('%UV_ZIP%','%UV_DIR%', $true)"
if errorlevel 1 goto :install_uv_extract_failed

:install_uv_post_extract
del /f /q "%UV_ZIP%" >nul 2>&1

if not exist "%UV_BIN%" goto :install_uv_missing_bin

endlocal
exit /b 0

:install_uv_unsupported_arch
echo Error: unsupported Windows architecture '%ARCH%'.>&2
endlocal
exit /b 1

:install_uv_download_failed
echo Error: failed to download uv zip.>&2
endlocal
exit /b 1

:install_uv_extract_failed
echo Error: failed to extract uv zip.>&2
endlocal
exit /b 1

:install_uv_missing_bin
echo Error: uv extracted but '%UV_BIN%' is missing.>&2
endlocal
exit /b 1
