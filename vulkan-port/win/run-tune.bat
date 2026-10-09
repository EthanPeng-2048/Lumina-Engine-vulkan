@echo off
REM Build + run the tunable strip-GEMV sweep on the NV device (single-submit timing).
call "%~dp0env.bat" || exit /b 1

cmake -S "%~dp0.." -B "%~dp0..\build" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_PREFIX_PATH=%VULKAN_SDK% >nul || exit /b 1
cmake --build "%~dp0..\build" -j %NUMBER_OF_PROCESSORS% || exit /b 1

cd /d "%~dp0..\build" || exit /b 1
lumina_q1.exe . --device 1 --rows 4096 --cols 14336 --iters 1000 --barrier
exit /b %errorlevel%
