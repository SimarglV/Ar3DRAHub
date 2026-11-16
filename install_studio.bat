@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul

echo.
echo ================================================
echo   Ar3DRA Hub - studio installer (Windows)
echo ================================================
echo.
echo Python 3.11 + the studio's packages. No model weights.
echo uv is installed first if missing (it asks first).

powershell -ExecutionPolicy ByPass -File "%~dp0install_studio.ps1"
if errorlevel 1 (
    echo.
    echo [ERROR] Studio setup failed. See the messages above.
    pause
    exit /b 1
)
pause
exit /b 0
