@echo off
REM Lumina-Engine Vulkan port - configure + build (Windows / MSVC / Ninja)
call "%~dp0env.bat" || exit /b 1

cd /d "%LUMINA_ROOT%"
echo [build] source = %LUMINA_ROOT%
echo [build] out    = %LUMINA_BUILD%

cmake -S . -B build-vulkan -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_PREFIX_PATH=%VULKAN_SDK% ^
  -DGGML_VULKAN=ON ^
  -DGGML_BACKEND_DL=OFF ^
  -DGGML_CPU_ALL_VARIANTS=OFF ^
  -DGGML_NATIVE=ON ^
  -DBUILD_SHARED_LIBS=OFF ^
  -DLLAMA_BUILD_TESTS=ON ^
  -DLLAMA_BUILD_EXAMPLES=OFF ^
  -DLLAMA_BUILD_TOOLS=ON ^
  -DLLAMA_BUILD_SERVER=OFF ^
  -DLLAMA_CURL=OFF || exit /b 1

cmake --build build-vulkan --target llama-cli llama-bench test-backend-ops -j %NUMBER_OF_PROCESSORS% || exit /b 1

echo [build] done.
exit /b 0
