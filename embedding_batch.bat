@echo off
title TPL 인수증 바코드 스캔 데몬

:: 작업 디렉토리 이동
cd /d "c:\workspace2

echo ==========================================================
echo    임베딩 프로세스 기동
echo ==========================================================
echo.

start "" ".\venv\Scripts\python.exe" embedding_batch.py