@echo off
call "D:\VS2022\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=C:\Program Files\CMake\bin;D:\VS2022\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%
cd /d D:\Lumina-Engine\llama.cpp-b10991
cmake -S . -B build-release -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DBUILD_SHARED_LIBS=ON ^
  -DGGML_BACKEND_DL=ON ^
  -DGGML_CPU_ALL_VARIANTS=ON ^
  -DGGML_NATIVE=OFF ^
  -DGGML_RPC=ON ^
  -DLLAMA_BUILD_TESTS=OFF ^
  -DLLAMA_BUILD_EXAMPLES=OFF ^
  -DLLAMA_BUILD_TOOLS=ON ^
  -DLLAMA_BUILD_SERVER=ON ^
  -DLLAMA_BUILD_APP=OFF ^
  -DLLAMA_CURL=OFF ^
  %*
