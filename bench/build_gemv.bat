@echo off
call "D:\VS2022\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\Lumina-Engine\bench
cl /nologo /O2 /arch:AVX2 /EHsc /Fe:bench_gemv.exe bench_gemv.cpp
