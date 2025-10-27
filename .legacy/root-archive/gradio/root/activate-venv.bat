@echo off
REM Workspace root = pasta deste .bat
set "ROOT=%~dp0"
set "VENV=%ROOT%.venv"
set "ACT_CMD=%VENV%\Scripts\activate.bat"
set "ACT_PS=%VENV%\Scripts\Activate.ps1"

if not exist "%VENV%" (
  echo ERRO: Nao encontrei "%VENV%".
  exit /b 1
)

REM Detecta se este cmd foi iniciado como "cmd /c ..." (tipico quando vc executa .bat a partir do pwsh)
set "CMDLINE=%CMDCMDLINE%"
set "CMDLINE_NO_C=%CMDLINE:/c=%"
if /i not "%CMDLINE%"=="%CMDLINE_NO_C%" goto LaunchedByOther

REM Modo 1: voce ja esta em um CMD interativo -> ativa na propria janela
if exist "%ACT_CMD%" (
  call "%ACT_CMD%"
  goto :eof
) else (
  echo ERRO: Nao encontrei "%ACT_CMD%".
  exit /b 1
)

:LaunchedByOther
REM Modo 2: script foi disparado por outro host (ex.: pwsh). Abrimos um PowerShell com o venv ativo.
if exist "%ACT_PS%" (
  where pwsh >nul 2>nul
  if %errorlevel%==0 (
    pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -NoExit -File "%ACT_PS%"
    goto :eof
  )
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -NoExit -File "%ACT_PS%"
  goto :eof
)

REM Fallback: sem Activate.ps1? Entao abrimos um CMD com o venv ativo.
if exist "%ACT_CMD%" (
  start "" cmd /K "call ""%ACT_CMD%"""
  goto :eof
) else (
  echo ERRO: Nao encontrei nenhum ativador: "%ACT_PS%" nem "%ACT_CMD%".
  exit /b 1
)