@echo off
call "D:\VS2022\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=D:\llvm\bin;%PATH%
cd /d D:\Lumina-Engine\bench
clang++ -O3 -DNDEBUG -mf16c -mfma -mavx -mavx2 bench_gemv128.cpp -o bench_gemv128_avx2.exe
clang++ -O3 -DNDEBUG -mf16c -mfma -mavx -mavx2 -mavxvnni bench_gemv128.cpp -o bench_gemv128_vnni.exe
