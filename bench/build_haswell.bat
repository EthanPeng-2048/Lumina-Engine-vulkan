@echo off
call "D:\VS2022\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=D:\VS2022\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%
cd /d D:\Lumina-Engine\llama.cpp-b10991
cmake --build build-release --target ggml-cpu-haswell -j 6
