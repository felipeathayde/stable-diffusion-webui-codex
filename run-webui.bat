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
    echo Run: install-webui.bat>&2
    exit /b 1
)

if defined PYTHONPATH (
    set "PYTHONPATH=%ROOT%;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%ROOT%"
)

set "SCRIPT=%ROOT%apps\codex_launcher.py"
set "CODEX_ROOT=%ROOT%"

"%PY_BIN%" "%SCRIPT%" %*
