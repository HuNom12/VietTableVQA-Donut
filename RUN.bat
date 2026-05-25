@echo off
cd /d "%~dp0"
title TableVQA
color 0B
cls

echo ===================================================
echo   KIEM TRA HE THONG
echo ===================================================
echo.

:: 1. Kiem tra Docker co cai tren may chua
where docker >nul 2>&1
if %errorlevel% neq 0 (
    echo [LOI] Khong tim thay Docker
    pause
    exit
)

:: 2. Kiem tra Engine truoc
echo [*] Dang kiem tra Engine...
docker info >nul 2>&1
if %errorlevel% == 0 goto :DOCKER_READY

:: 3. Neu Engine chua Ready, cu "Bam coi" cho Docker hien len
echo [!] Docker Engine chua san sang. Dang ra lenh kich hoat...
if exist "C:\Program Files\Docker\Docker\Docker Desktop.exe" (
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
) else (
    echo [!] Khong tim thay file thuc thi. Nam tu bat Docker bang tay nhe!
)

echo.
echo Dang bat Docker...
echo.

:WAIT_LOOP
<nul set /p=.
timeout /t 5 >nul
docker info >nul 2>&1
if %errorlevel% neq 0 goto :WAIT_LOOP

:DOCKER_READY
echo.
echo [OK] Docker san sang.
echo.
echo ===================================================
echo   DANG CHAY DOCKER-COMPOSE...
echo ===================================================
docker-compose up -d

echo.
echo [*] Dang load Model...
timeout /t 20 /nobreak > nul

echo [*] MO APP: http://localhost:8501
start chrome --app=http://localhost:8501
