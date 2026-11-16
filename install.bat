@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul

echo.
echo ================================================
echo   UniMate - one-command installer
echo ================================================
echo.

REM ── 1. Make sure uv is available (it fetches Python 3.10) ──────────────
where uv >nul 2>&1
if not errorlevel 1 goto :have_uv

echo uv is needed to set up Python 3.10: Astral's Python installer, ~40 MB.
set /p "ANSWER=Install uv from astral.sh? [y/N] "
if /i not "%ANSWER%"=="y" (
    echo Stopped. Nothing was changed.
    pause
    exit /b 1
)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uv installed, but this window cannot see it yet.
    echo         Open a new Command Prompt and run this again.
    pause
    exit /b 1
)

:have_uv

REM ── 2. Create the virtual environment (uv fetches Python 3.10) ─────────
echo.
echo [1/4] Creating virtual environment (.venv, Python 3.10) ...
if not exist ".venv\Scripts\python.exe" goto :make_venv
".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,10) else 1)" >nul 2>&1
if errorlevel 1 goto :replace_venv
echo         Already present: .venv (Python 3.10)
goto :have_venv

:replace_venv
echo         Replacing .venv: it is not Python 3.10.
uv venv --clear --seed --no-project --python 3.10 .venv
if errorlevel 1 goto :err
goto :have_venv

:make_venv
uv venv --seed --no-project --python 3.10 .venv
if errorlevel 1 goto :err

:have_venv

REM ── 3. Install dependencies ────────────────────────────────────────────
echo.
echo [2/4] Installing setuptools^<81 ...
".venv\Scripts\python.exe" -m pip install --upgrade pip "setuptools<81"
if errorlevel 1 goto :err

echo.
echo [3/4] Installing dependencies (this can take several minutes) ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt --no-build-isolation
if errorlevel 1 goto :err

REM ── 4. Download the checkpoint ─────────────────────────────────────────
set "CKPT=outputs\unimate_uniml3d_f60_v3\checkpoints\checkpoint_step_150000.pt"
echo.
echo [4/4] Downloading the model checkpoint ...
if exist "%CKPT%" (
    echo         Already present: %CKPT%
) else (
    ".venv\Scripts\hf.exe" download Linzhan/UniMate --repo-type model --local-dir outputs ^
        --include "unimate_uniml3d_f60_v3/*.json" ^
        --include "unimate_uniml3d_f60_v3/*.npy" ^
        --include "unimate_uniml3d_f60_v3/checkpoints/checkpoint_step_150000.pt"
    if errorlevel 1 goto :err
)

echo.
echo ================================================
echo   Done.
echo   UniMate's environment is ready.
echo   Next: install the studio (install_studio.ps1), then start it with:
echo     .\lab          (or  run.bat)
echo ================================================
pause
exit /b 0

:err
echo.
echo [ERROR] Installation failed. See the messages above.
pause
exit /b 1
