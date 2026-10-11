@echo off
REM Run the Q1_0 Vulkan port harness.
REM   run-port.bat [--device N] [--rows M] [--cols K] [--iters N]
set "BIN=%~dp0..\build\lumina_q1.exe"
if not exist "%BIN%" (
  echo [run-port] %BIN% not found - run build-port.bat first
  exit /b 1
)
"%BIN%" "%~dp0..\build" %*
