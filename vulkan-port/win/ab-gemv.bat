@echo off
REM A/B the Q1_0 GEMV path in ggml's own perf tool: on (default) vs off (native float path).
cd /d C:\Users\ethan\Codes\lumina-vulkan\llama.cpp-b10991\build-vulkan\bin || exit /b 1
echo === GEMV default (on) ===
test-backend-ops.exe perf -b Vulkan1 -o MUL_MAT -p q1_0 | findstr /C:"n=1,"
echo === GEMV off (LUMINA_Q1_VK_GEMV=0) ===
set LUMINA_Q1_VK_GEMV=0
test-backend-ops.exe perf -b Vulkan1 -o MUL_MAT -p q1_0 | findstr /C:"n=1,"
exit /b 0
