@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "UV_BIN=%ROOT%\.uv\bin\uv.exe"
set "PYTHON_BIN=%ROOT%\.venv\Scripts\python.exe"
set "NPM_BIN=%ROOT%\.nodeenv\Scripts\npm.cmd"
set "INTERFACE_DIR=%ROOT%\apps\interface"
set "PACKAGE_LOCK=%INTERFACE_DIR%\package-lock.json"
set "CODEX_FFMPEG_VERSION=%CODEX_FFMPEG_VERSION%"
if "%CODEX_FFMPEG_VERSION%"=="" set "CODEX_FFMPEG_VERSION=7.0.2"
set "ALLOW_UNTRACKED=0"

:parse_args
if "%~1"=="" goto :args_done
if /I "%~1"=="--help" goto :usage
if /I "%~1"=="-h" goto :usage
if /I "%~1"=="--force" (
  set "ALLOW_UNTRACKED=1"
  shift
  goto :parse_args
)
if /I "%~1"=="-f" (
  set "ALLOW_UNTRACKED=1"
  shift
  goto :parse_args
)
call :die E_BAD_ARGS "Unknown argument '%~1'. Use --help."
exit /b 1

:args_done

call :validate_git_state
if errorlevel 1 exit /b 1
if /I "%ALLOW_UNTRACKED%"=="1" call :log "Force mode enabled: untracked paths are ignored in dirty check."

for /f "usebackq delims=" %%I in (`git -C "%ROOT%" rev-parse HEAD 2^>nul`) do set "HEAD_BEFORE=%%I"
if not defined HEAD_BEFORE call :die E_HEAD_UNRESOLVED "Failed to resolve current HEAD."
if errorlevel 1 exit /b 1

call :log "Fetching upstream refs ..."
git -C "%ROOT%" fetch --prune
if errorlevel 1 call :die E_FETCH_FAILED "git fetch --prune failed."
if errorlevel 1 exit /b 1

set "AHEAD="
set "BEHIND="
for /f "tokens=1,2" %%A in ('git -C "%ROOT%" rev-list --left-right --count HEAD...@{u} 2^>nul') do (
  set "AHEAD=%%A"
  set "BEHIND=%%B"
)
if not defined AHEAD call :die E_UPSTREAM_COUNT_FAILED "Failed to compute ahead/behind status."
if errorlevel 1 exit /b 1
if not defined BEHIND call :die E_UPSTREAM_COUNT_FAILED "Failed to compute ahead/behind status."
if errorlevel 1 exit /b 1

if %AHEAD% GTR 0 if %BEHIND% GTR 0 call :die E_DIVERGED "Branch diverged from upstream (ahead=%AHEAD%, behind=%BEHIND%). Reconcile manually first."
if errorlevel 1 exit /b 1
if %AHEAD% GTR 0 call :die E_AHEAD_OF_UPSTREAM "Local branch is ahead by %AHEAD% commit(s). Push/rebase before update."
if errorlevel 1 exit /b 1
call :prepare_refresh_requirements
if errorlevel 1 exit /b 1
call :resolve_torch_backend
if errorlevel 1 exit /b 1

call :log "Resolved torch backend extra: %TORCH_BACKEND%"
if %BEHIND% EQU 0 (
  call :log "Already up to date. No commits pulled; running environment refresh."
) else (
  call :log "Pulling updates (ff-only) ..."
  git -C "%ROOT%" pull --ff-only
  if errorlevel 1 call :die E_PULL_FAILED "git pull --ff-only failed."
  if errorlevel 1 exit /b 1

  for /f "usebackq delims=" %%I in (`git -C "%ROOT%" rev-parse HEAD 2^>nul`) do set "HEAD_AFTER=%%I"
  if not defined HEAD_AFTER call :die E_HEAD_UNRESOLVED "Failed to resolve HEAD after pull."
  if errorlevel 1 exit /b 1
  if /I "%HEAD_AFTER%"=="%HEAD_BEFORE%" (
    call :log "No commit change after pull. Running environment refresh anyway."
  ) else (
    call :log "Pulled new commits from upstream."
  )
)

call :refresh_environment
if errorlevel 1 exit /b 1

call :log "Update completed successfully."
exit /b 0

:usage
echo Usage: update-webui.bat [--force] [--help]
echo.
echo Safe updater for stable-diffusion-webui-codex.
echo.
echo Behavior:
echo   - Fail-closed preflight ^(dirty tree, detached HEAD, no upstream, ahead/diverged, git operation in progress^).
echo   - No destructive commands ^(no reset/clean/restore/delete of user files^).
echo   - Git update only via: fetch --prune ^+ pull --ff-only.
echo   - Dependency verification runs on every update attempt after git safety checks.
echo   - Environment refresh runs on every update attempt after dependency verification.
echo.
echo Policy:
echo   - Scope: repo root only ^(no submodule/extension updates^).
echo   - --force disables untracked-path preflight checks only; tracked changes still abort and git pull safety still applies.
echo   - Ignored paths do not block update ^(P2=B^).
echo   - Frontend refresh uses lock-preserving mode: npm ci.
echo.
echo Optional env:
echo   CODEX_TORCH_BACKEND=cpu^|cu126^|cu128^|cu130^|rocm64
echo   CODEX_FFMPEG_VERSION=^<version^>  ^(default 7.0.2^)
exit /b 0

:log
echo [update] %~1
exit /b 0

:die
echo [update][%~1] %~2 1>&2
exit /b 1

:validate_git_state
where git >nul 2>nul
if errorlevel 1 call :die E_TOOL_MISSING "Required command 'git' not found."
if errorlevel 1 exit /b 1

set "INSIDE="
for /f "usebackq delims=" %%I in (`git -C "%ROOT%" rev-parse --is-inside-work-tree 2^>nul`) do set "INSIDE=%%I"
if /I not "%INSIDE%"=="true" call :die E_NOT_GIT_REPO "Script root '%ROOT%' is not inside a git worktree."
if errorlevel 1 exit /b 1

for %%I in ("%ROOT%") do set "ROOT_CANON=%%~fI"
set "TOPLEVEL="
for /f "usebackq delims=" %%I in (`git -C "%ROOT%" rev-parse --show-toplevel 2^>nul`) do set "TOPLEVEL=%%~fI"
if not defined TOPLEVEL call :die E_GIT_TOPLEVEL_UNRESOLVED "Failed to resolve git top-level."
if errorlevel 1 exit /b 1
if /I not "%TOPLEVEL%"=="%ROOT_CANON%" call :die E_WRONG_REPO_ROOT "Run updater from repository root '%ROOT_CANON%' only."
if errorlevel 1 exit /b 1

set "BRANCH="
for /f "usebackq delims=" %%I in (`git -C "%ROOT%" symbolic-ref --quiet --short HEAD 2^>nul`) do set "BRANCH=%%I"
if not defined BRANCH call :die E_DETACHED_HEAD "Detached HEAD detected. Checkout a branch and rerun."
if errorlevel 1 exit /b 1

set "UPSTREAM="
for /f "usebackq delims=" %%I in (`git -C "%ROOT%" rev-parse --abbrev-ref --symbolic-full-name "@{u}" 2^>nul`) do set "UPSTREAM=%%I"
if not defined UPSTREAM call :die E_NO_UPSTREAM "No upstream configured for branch '%BRANCH%'."
if errorlevel 1 exit /b 1

set "GIT_DIR_RAW="
for /f "usebackq delims=" %%I in (`git -C "%ROOT%" rev-parse --git-dir 2^>nul`) do set "GIT_DIR_RAW=%%I"
if not defined GIT_DIR_RAW call :die E_GIT_DIR_UNRESOLVED "Failed to resolve git dir."
if errorlevel 1 exit /b 1

set "GIT_DIR=%GIT_DIR_RAW%"
if not "%GIT_DIR_RAW:~0,1%"=="\" if not "%GIT_DIR_RAW:~1,1%"==":" set "GIT_DIR=%ROOT%\%GIT_DIR_RAW%"

if exist "%GIT_DIR%\MERGE_HEAD" call :die E_GIT_MERGE_IN_PROGRESS "Merge in progress. Resolve or abort merge before update."
if errorlevel 1 exit /b 1
if exist "%GIT_DIR%\CHERRY_PICK_HEAD" call :die E_GIT_CHERRY_PICK_IN_PROGRESS "Cherry-pick in progress. Resolve or abort before update."
if errorlevel 1 exit /b 1
if exist "%GIT_DIR%\REVERT_HEAD" call :die E_GIT_REVERT_IN_PROGRESS "Revert in progress. Resolve or abort before update."
if errorlevel 1 exit /b 1
if exist "%GIT_DIR%\rebase-apply\NUL" call :die E_GIT_REBASE_IN_PROGRESS "Rebase in progress. Complete or abort rebase before update."
if errorlevel 1 exit /b 1
if exist "%GIT_DIR%\rebase-merge\NUL" call :die E_GIT_REBASE_IN_PROGRESS "Rebase in progress. Complete or abort rebase before update."
if errorlevel 1 exit /b 1
if exist "%GIT_DIR%\BISECT_LOG" call :die E_GIT_BISECT_IN_PROGRESS "Bisect in progress. Finish bisect before update."
if errorlevel 1 exit /b 1

set "TMP_STATUS=%TEMP%\codex-update-status-%RANDOM%-%RANDOM%.txt"
set "TMP_TRACKED=%TEMP%\codex-update-tracked-%RANDOM%-%RANDOM%.txt"
set "TMP_UNTRACKED=%TEMP%\codex-update-untracked-%RANDOM%-%RANDOM%.txt"

set "STATUS_UNTRACKED_MODE=all"
if /I "%ALLOW_UNTRACKED%"=="1" set "STATUS_UNTRACKED_MODE=no"
git -C "%ROOT%" status --porcelain=v1 --untracked-files=%STATUS_UNTRACKED_MODE% > "%TMP_STATUS%"
if errorlevel 1 (
  call :delete_if_exists "%TMP_STATUS%"
  call :die E_STATUS_FAILED "Failed to inspect git status."
  exit /b 1
)

for %%I in ("%TMP_STATUS%") do set "STATUS_SIZE=%%~zI"
if not defined STATUS_SIZE set "STATUS_SIZE=0"

if %STATUS_SIZE% GTR 0 (
  type nul > "%TMP_TRACKED%"
  type nul > "%TMP_UNTRACKED%"
  findstr /B /C:"?? " "%TMP_STATUS%" > "%TMP_UNTRACKED%" 2>nul
  findstr /V /B /C:"?? " "%TMP_STATUS%" > "%TMP_TRACKED%" 2>nul

  echo [update][E_WORKTREE_DIRTY] Local changes detected; update aborted to protect your files. 1>&2
  call :print_status_section "Tracked changes:" "%TMP_TRACKED%"
  if /I "%ALLOW_UNTRACKED%"=="1" (
    echo [update][E_WORKTREE_DIRTY] Untracked-path preflight checks were disabled by --force. 1>&2
    echo [update][E_WORKTREE_DIRTY] Remediation: commit/stash tracked changes, then rerun. 1>&2
  ) else (
    call :print_status_section "Untracked paths:" "%TMP_UNTRACKED%"
    echo [update][E_WORKTREE_DIRTY] Ignored paths are excluded by policy ^(P2=B^). 1>&2
    echo [update][E_WORKTREE_DIRTY] Remediation: commit/stash tracked changes and move or remove untracked paths, then rerun. 1>&2
  )

  call :delete_if_exists "%TMP_STATUS%"
  call :delete_if_exists "%TMP_TRACKED%"
  call :delete_if_exists "%TMP_UNTRACKED%"
  exit /b 1
)

call :delete_if_exists "%TMP_STATUS%"
call :delete_if_exists "%TMP_TRACKED%"
call :delete_if_exists "%TMP_UNTRACKED%"
exit /b 0

:print_status_section
echo [update][E_WORKTREE_DIRTY] %~1 1>&2
for %%I in ("%~2") do set "SECTION_SIZE=%%~zI"
if not defined SECTION_SIZE set "SECTION_SIZE=0"
if %SECTION_SIZE% LEQ 0 (
  echo   - ^(none^) 1>&2
  exit /b 0
)
for /f "usebackq delims=" %%L in ("%~2") do echo   - %%L 1>&2
exit /b 0

:delete_if_exists
set "TARGET=%~f1"
if not defined TARGET exit /b 0
for %%I in ("%TARGET%") do (
  set "TARGET_NAME=%%~nxI"
  set "TARGET_DIR=%%~dpI"
)
set "TEMP_DIR="
for %%I in ("%TEMP%\.") do set "TEMP_DIR=%%~fI"
if not defined TEMP_DIR exit /b 0
if /I not "%TARGET_DIR%"=="%TEMP_DIR%" exit /b 0
echo %TARGET_NAME%| findstr /B /C:"codex-update-" >nul 2>nul
if errorlevel 1 exit /b 0
if exist "%TARGET%" del /q "%TARGET%" >nul 2>nul
exit /b 0

:prepare_refresh_requirements
if not exist "%UV_BIN%" call :die E_UV_MISSING "uv not found at '%UV_BIN%'. Run install-webui.bat first."
if errorlevel 1 exit /b 1
if not exist "%PYTHON_BIN%" call :die E_PYTHON_MISSING "Python runtime missing at '%PYTHON_BIN%'. Run install-webui.bat first."
if errorlevel 1 exit /b 1
if not exist "%NPM_BIN%" call :die E_NPM_MISSING "npm not found at '%NPM_BIN%'. Run install-webui.bat first."
if errorlevel 1 exit /b 1
if not exist "%PACKAGE_LOCK%" call :die E_NPM_LOCK_MISSING "Lock-preserving update requires '%PACKAGE_LOCK%'."
if errorlevel 1 exit /b 1
exit /b 0

:resolve_torch_backend
set "TORCH_BACKEND="
if defined CODEX_TORCH_BACKEND (
  if /I "%CODEX_TORCH_BACKEND%"=="cpu" set "TORCH_BACKEND=cpu"
  if /I "%CODEX_TORCH_BACKEND%"=="cu126" set "TORCH_BACKEND=cu126"
  if /I "%CODEX_TORCH_BACKEND%"=="cu128" set "TORCH_BACKEND=cu128"
  if /I "%CODEX_TORCH_BACKEND%"=="cu130" set "TORCH_BACKEND=cu130"
  if /I "%CODEX_TORCH_BACKEND%"=="rocm64" set "TORCH_BACKEND=rocm64"
  if not defined TORCH_BACKEND call :die E_INVALID_TORCH_BACKEND "Invalid CODEX_TORCH_BACKEND='%CODEX_TORCH_BACKEND%'."
  if errorlevel 1 exit /b 1
  exit /b 0
)

for /f "usebackq delims=" %%I in (`"%PYTHON_BIN%" -c "import sys,importlib.util; spec=importlib.util.find_spec('torch'); spec or sys.exit(2); import torch; v=str(getattr(torch,'__version__','')).lower(); m=[('+cu126','cu126'),('+cu128','cu128'),('+cu130','cu130'),('+rocm','rocm64'),('+cpu','cpu')]; r=next((name for token,name in m if token in v),None); r and print(r) or sys.exit(3)" 2^>nul`) do set "TORCH_BACKEND=%%I"

if not defined TORCH_BACKEND call :die E_TORCH_BACKEND_UNRESOLVED "Could not detect installed torch backend extra. Set CODEX_TORCH_BACKEND explicitly."
if errorlevel 1 exit /b 1
exit /b 0

:refresh_environment
set "CODEX_ROOT=%ROOT%"
if defined PYTHONPATH (
  set "PYTHONPATH=%ROOT%;%PYTHONPATH%"
) else (
  set "PYTHONPATH=%ROOT%"
)

call :log "Refreshing Python dependencies with backend extra '%TORCH_BACKEND%' ..."
"%UV_BIN%" sync --locked --extra "%TORCH_BACKEND%"
if errorlevel 1 call :die E_UV_SYNC_FAILED "uv sync failed."
if errorlevel 1 exit /b 1

call :log "Refreshing runtime assets (ffmpeg/ffprobe + RIFE model) ..."
"%PYTHON_BIN%" -c "import os,sys; from apps.backend.video.runtime_dependencies import ensure_ffmpeg_binaries, ensure_rife_model_file; v=os.environ.get('CODEX_FFMPEG_VERSION') or '7.0.2'; b=ensure_ffmpeg_binaries(version=v); m=ensure_rife_model_file(); print('[update] ffmpeg: ' + str(b['ffmpeg'])); print('[update] ffprobe: ' + str(b['ffprobe'])); print('[update] RIFE model: ' + str(m))"
if errorlevel 1 call :die E_RUNTIME_PROVISION_FAILED "Runtime dependency provisioning failed."
if errorlevel 1 exit /b 1

call :log "Refreshing frontend dependencies with npm ci ..."
pushd "%INTERFACE_DIR%" >nul 2>nul
if errorlevel 1 call :die E_INTERFACE_DIR_MISSING "Interface directory not found: '%INTERFACE_DIR%'."
if errorlevel 1 exit /b 1
"%NPM_BIN%" ci --no-audit --no-fund
set "NPM_EXIT=%ERRORLEVEL%"
popd >nul 2>nul
if not "%NPM_EXIT%"=="0" call :die E_NPM_CI_FAILED "npm ci failed."
if not "%NPM_EXIT%"=="0" exit /b 1
exit /b 0
