@echo off
setlocal ENABLEDELAYEDEXPANSION
REM Launch the BIOS-style TUI on Windows
REM Looks for a virtualenv at .venv; falls back to system Python

set ROOT=%~dp0
set VENV=%ROOT%\.venv

if exist "%VENV%\Scripts\python.exe" (
  echo Using venv Python at %VENV%\Scripts\python.exe
  "%VENV%\Scripts\python.exe" -u tools\tui_bios.py %*
) else (
  where py >NUL 2>&1
  if %ERRORLEVEL%==0 (
    echo Using 'py' launcher
    py -3 -u tools\tui_bios.py %*
  ) else (
    echo Using system 'python'
    python -u tools\tui_bios.py %*
  )
)

endlocal
