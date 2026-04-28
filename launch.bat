@echo off
REM ─────────────────────────────────────────────────────────────
REM  DeepGuard / Sach-AI  — Quick Launch Script
REM ─────────────────────────────────────────────────────────────

REM Ensure we are in the directory where the script is located
cd /d %~dp0

:menu
cls
echo.
echo  ==========================================================
echo   🛡️ DeepGuard / Sach-AI  ^|  Multimodal Detection
echo  ==========================================================
echo.
echo  [1] 🚀 Quick Launch UI (Direct)
echo  [2] 🌐 Launch REST API (FastAPI)
echo  [3] 🔍 System Diagnostic
echo  [4] 🏋️  Start Training
echo  [5] 🚪 Exit
echo.
set /p choice=Select option (1-5): 

if "%choice%"=="1" goto opt1
if "%choice%"=="2" goto opt2
if "%choice%"=="3" goto opt3
if "%choice%"=="4" goto opt4
if "%choice%"=="5" exit
goto menu

:opt1
echo.
echo Starting Web Dashboard...
python main.py ui
pause
goto menu

:opt2
echo.
echo Starting API Server...
python main.py api
pause
goto menu

:opt3
echo.
python main.py check
pause
goto menu

:opt4
echo.
set /p mod=Modality (video/audio/image/all): 
python main.py train --modality %mod%
pause
goto menu
