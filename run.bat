@echo off
rem TableDrawer 启动脚本：优先使用项目自带的 .venv，找不到则回退到系统 python
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

"%PY%" app.py %*
if errorlevel 1 (
    echo.
    echo [提示] 程序异常退出，错误码 %errorlevel%
    pause
)
endlocal
