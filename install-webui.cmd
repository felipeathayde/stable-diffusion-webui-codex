@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
set "PS1=%ROOT%install-webui.ps1"

if not exist "%PS1%" (
  echo Error: missing "%PS1%".>&2
  exit /b 1
)

where pwsh >nul 2>nul
if not errorlevel 1 goto :run_pwsh

where powershell >nul 2>nul
if not errorlevel 1 goto :run_powershell

echo Error: neither pwsh nor powershell is available on PATH.>&2
exit /b 1

:run_pwsh
pwsh -NoProfile -ExecutionPolicy Bypass -File "%PS1%" %*
set "CODEX_EXIT=%ERRORLEVEL%"
exit /b %CODEX_EXIT%

:run_powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%" %*
set "CODEX_EXIT=%ERRORLEVEL%"
exit /b %CODEX_EXIT%
