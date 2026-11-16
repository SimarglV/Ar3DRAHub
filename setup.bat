@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul

echo.
echo ================================================================
echo   Ar3DRA Hub - one-click installer (Windows)
echo ================================================================
echo.
echo This installs two environments side by side:
echo   1. .venv        - Python 3.10, UniMate motion. Downloads the
echo                     checkpoint (~6 GB) if it is missing.
echo   2. .venv-studio - Python 3.11, the studio. Packages only, no
echo                     model weights.
echo.
echo Needed: Python 3.10 (winget install Python.Python.3.10), an
echo NVIDIA GPU for generation, ~30 GB of free disk. Nothing else
echo downloads without asking.
echo.

REM /y skips the question (scripts and agents)
set "YES=0"
if /i "%~1"=="/y" set "YES=1"
if "%YES%"=="1" goto :go
set /p "ANSWER=Continue? [y/N] "
if /i not "%ANSWER%"=="y" (
    echo Stopped. Nothing was changed.
    pause
    exit /b 0
)

:go
echo.
echo ----------------------------------------------------------------
echo [1/2] UniMate: .venv + checkpoint
echo ----------------------------------------------------------------
call "%~dp0install.bat"
if errorlevel 1 goto :err

echo.
echo ----------------------------------------------------------------
echo [2/2] Studio: .venv-studio
echo ----------------------------------------------------------------
powershell -ExecutionPolicy ByPass -File "%~dp0install_studio.ps1"
if errorlevel 1 goto :err

echo.
echo ================================================================
echo   Done. Start the studio with:
echo     .\lab          (run.bat does the same)
echo   Then open Setup ^& Status: set up Pixal3D, and install Blender
echo   4.2+ yourself if you want Finish (the low-poly clean-up).
echo ================================================================
pause
exit /b 0

:err
echo.
echo [ERROR] Setup failed. See the messages above.
pause
exit /b 1
