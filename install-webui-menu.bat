@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"
pushd "%ROOT%" >nul

REM Interactive wrapper around install-webui.bat to avoid forcing users to set env vars manually.

:menu
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

if "%CHOICE%"=="1" goto :simple
if "%CHOICE%"=="2" goto :advanced
if "%CHOICE%"=="3" goto :done
goto :menu

:simple
echo.
echo [menu] Running simple install (auto-detect)...
echo.
call "%ROOT%install-webui.bat"
goto :post

:advanced
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

if "%BCHOICE%"=="5" goto :menu

REM Reset only what this wrapper owns.
set "CODEX_TORCH_MODE="
set "CODEX_TORCH_BACKEND="
set "CODEX_CUDA_VARIANT="

if "%BCHOICE%"=="1" (
  set "CODEX_TORCH_MODE=auto"
  goto :run_install
)
if "%BCHOICE%"=="2" (
  set "CODEX_TORCH_MODE=cpu"
  goto :run_install
)
if "%BCHOICE%"=="3" goto :cuda_menu
if "%BCHOICE%"=="4" (
  set "CODEX_TORCH_MODE=skip"
  goto :run_install
)
goto :advanced

:cuda_menu
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

if "%CCHOICE%"=="4" goto :advanced

set "CODEX_TORCH_MODE=cuda"
if "%CCHOICE%"=="1" set "CODEX_CUDA_VARIANT=12.6"
if "%CCHOICE%"=="2" set "CODEX_CUDA_VARIANT=12.8"
if "%CCHOICE%"=="3" set "CODEX_CUDA_VARIANT=13"

if "%CODEX_CUDA_VARIANT%"=="" goto :cuda_menu

goto :run_install

:run_install
echo.
echo [menu] Selected:
if not "%CODEX_TORCH_MODE%"=="" echo   - CODEX_TORCH_MODE=%CODEX_TORCH_MODE%
if not "%CODEX_TORCH_BACKEND%"=="" echo   - CODEX_TORCH_BACKEND=%CODEX_TORCH_BACKEND%
if not "%CODEX_CUDA_VARIANT%"=="" echo   - CODEX_CUDA_VARIANT=%CODEX_CUDA_VARIANT%
echo.
call "%ROOT%install-webui.bat"
goto :post

:post
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
if "%PCHOICE%"=="1" goto :menu
goto :done

:done
popd >nul
endlocal
exit /b 0

