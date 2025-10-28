@echo off
setlocal

set "ROOT=%~dp0"
if defined PYTHON (
    set "PY_BIN=%PYTHON%"
) else (
    set "PY_BIN=python"
)

where "%PY_BIN%" >nul 2>&1
if errorlevel 1 (
    echo Error: unable to locate a Python interpreter. Set PYTHON environment variable to a valid executable.>&2
    exit /b 1
)

set "SCRIPT=%ROOT%tools\tui_bios.py"

"%PY_BIN%" "%SCRIPT%" %*
