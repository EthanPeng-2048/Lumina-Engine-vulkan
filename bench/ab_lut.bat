@echo off
cd /d D:\Lumina-Engine\Lumina-Engine-Bin
set THREADS=%1
if "%THREADS%"=="" set THREADS=6
for /L %%i in (1,1,3) do (
  echo --- run %%i LUT OFF ---
  set GGML_Q1_LUT=0
  Lumina-Engine-bench.exe -m ..\Bonsai-8B.gguf -p 64 -n 128 -t %THREADS% -r 2 2>nul | findstr /C:"pp64" /C:"tg128"
  echo --- run %%i LUT ON (default) ---
  set GGML_Q1_LUT=
  Lumina-Engine-bench.exe -m ..\Bonsai-8B.gguf -p 64 -n 128 -t %THREADS% -r 2 2>nul | findstr /C:"pp64" /C:"tg128"
)
