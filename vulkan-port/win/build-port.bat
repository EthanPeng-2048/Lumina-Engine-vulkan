@echo off
REM Build the Lumina-Engine Vulkan Q1_0 port harness (shaders + host).
call "%~dp0env.bat" || exit /b 1

cmake -S "%~dp0.." -B "%~dp0..\build" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_PREFIX_PATH=%VULKAN_SDK% || exit /b 1

cmake --build "%~dp0..\build" -j %NUMBER_OF_PROCESSORS% || exit /b 1
echo [build-port] done.
exit /b 0
