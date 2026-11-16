@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul

if not exist ".venv-studio\Scripts\python.exe" (
    echo [ERROR] Studio environment not found.
    echo         Run: install_studio.bat
    pause
    exit /b 1
)

echo.
echo Starting the MCP server so an agent (Claude Code, Codex, ...)
echo can drive the studio. Press Ctrl+C to stop.
echo.
echo Connect your agent with:
echo   claude mcp add unimate -- "%CD%\lab.cmd" mcp
echo.
".venv-studio\Scripts\python.exe" -m Ar3DRAHub.mcp_server %*
pause
exit /b 0
