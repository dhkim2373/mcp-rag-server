@echo off
title OpenWebUI FastAPI 데몬

:: 작업 디렉토리 이동
cd /d "c:\workspace2

echo ==========================================================
echo    OpenWebUI 서버 기동
echo ==========================================================
echo.

start "" ".\venv\Scripts\python.exe" main.py