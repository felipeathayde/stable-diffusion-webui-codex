@echo off
setlocal

set "ROOT=%~dp0"
set "VENV_PY=%ROOT%.venv\Scripts\python.exe"
if defined PYTHON (
    set "PY_BIN=%PYTHON%"
) else if exist "%VENV_PY%" (
    set "PY_BIN=%VENV_PY%"
) else (
    set "PY_BIN=python"
)

REM If PY_BIN is a full path, `where` will fail; prefer a direct existence check first.
if not exist "%PY_BIN%" (
    where "%PY_BIN%" >nul 2>&1
    if errorlevel 1 (
        echo Error: unable to locate a Python interpreter. Set PYTHON environment variable to a valid executable.>&2
        exit /b 1
    )
)

if defined PYTHONPATH (
    set "PYTHONPATH=%ROOT%;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%ROOT%"
)

set "SCRIPT=%ROOT%apps\tui_launcher.py"
set "CODEX_ROOT=%ROOT%"

"%PY_BIN%" "%SCRIPT%" %*
