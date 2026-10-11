@echo off
REM Build ggml, verify Q1_0 correctness, A/B the GEMV path, and probe the prefill paths.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=C:\Program Files\CMake\bin;C:\VulkanSDK\1.4.350.0\Bin;%PATH%"
cd /d C:\Users\ethan\Codes\lumina-vulkan\llama.cpp-b10991\build-vulkan || exit /b 1
cmake --build . --target test-backend-ops -j 8 >nul 2>&1
if errorlevel 1 (echo BUILD_FAILED & cmake --build . --target test-backend-ops -j 8 & exit /b 1)
echo BUILD_OK
cd bin || exit /b 1

echo === correctness (GEMV on by default) ===
test-backend-ops.exe test -b Vulkan1 -o MUL_MAT -p q1_0 | findstr /C:"tests passed"

echo === perf: GEMV on (default) ===
test-backend-ops.exe perf -b Vulkan1 -o MUL_MAT -p q1_0 | findstr /C:"n=1," /C:"n=512"

echo === perf: GEMV off (native) ===
set LUMINA_Q1_VK_GEMV=0
test-backend-ops.exe perf -b Vulkan1 -o MUL_MAT -p q1_0 | findstr /C:"n=1,"
set "LUMINA_Q1_VK_GEMV="

echo === prefill: disable coopmat2 (KHR / generic path) ===
set GGML_VK_DISABLE_COOPMAT2=1
test-backend-ops.exe perf -b Vulkan1 -o MUL_MAT -p q1_0 | findstr /C:"n=512"
set "GGML_VK_DISABLE_COOPMAT2="

echo === prefill: disable coopmat and coopmat2 (pure shared-staging mul_mm) ===
set GGML_VK_DISABLE_COOPMAT=1
set GGML_VK_DISABLE_COOPMAT2=1
test-backend-ops.exe perf -b Vulkan1 -o MUL_MAT -p q1_0 | findstr /C:"n=512"
exit /b 0
