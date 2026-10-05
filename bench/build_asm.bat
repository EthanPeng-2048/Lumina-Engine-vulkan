@echo off
call "D:\VS2022\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\Lumina-Engine\bench
cl /nologo /O2 /arch:AVX2 /EHsc /FAs /Fe:bench_q1p_asm.exe bench_q1p.cpp >nul
