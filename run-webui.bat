@echo off
setlocal

set "ROOT=%~dp0"
set "NODEENV=%ROOT%.nodeenv"
if exist "%NODEENV%\Scripts\node.exe" set "PATH=%NODEENV%\Scripts;%PATH%"
if exist "%NODEENV%\bin\node.exe" set "PATH=%NODEENV%\bin;%PATH%"
set "VENV_PY=%ROOT%.venv\Scripts\python.exe"
if defined PYTHON (
    set "PY_BIN=%PYTHON%"
) else (
    set "PY_BIN=%VENV_PY%"
)

if not exist "%PY_BIN%" (
    echo Error: expected Python at "%PY_BIN%".>&2
    echo Run: install-webui.cmd>&2
    exit /b 1
)

if defined PYTHONPATH (
    set "PYTHONPATH=%ROOT%;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%ROOT%"
)

if defined PYTORCH_CUDA_ALLOC_CONF (
    echo Error: legacy env var PYTORCH_CUDA_ALLOC_CONF is no longer supported.>&2
    echo Found value: "%PYTORCH_CUDA_ALLOC_CONF%".>&2
    echo Use PYTORCH_ALLOC_CONF instead.>&2
    echo.>&2
    echo To clear it for this current cmd.exe session:>&2
    echo   set PYTORCH_CUDA_ALLOC_CONF=>&2
    echo.>&2
    echo To clear it for this current PowerShell session:>&2
    echo   Remove-Item Env:PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue>&2
    echo.>&2
    echo To remove a persisted User/Machine variable in PowerShell:>&2
    echo   [Environment]::SetEnvironmentVariable('PYTORCH_CUDA_ALLOC_CONF',$null,'User')>&2
    echo   [Environment]::SetEnvironmentVariable('PYTORCH_CUDA_ALLOC_CONF',$null,'Machine')>&2
    exit /b 2
)

set "SCRIPT=%ROOT%apps\codex_launcher.py"
set "CODEX_ROOT=%ROOT%"

"%PY_BIN%" "%SCRIPT%" %*
