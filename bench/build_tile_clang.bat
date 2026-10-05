@echo off
call "D:\VS2022\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=D:\llvm\bin;%PATH%
cd /d D:\Lumina-Engine\bench
clang++ -O3 -DNDEBUG -msse4.2 -mf16c -mfma -mbmi2 -mavx -mavx2 bench_tile.cpp -o bench_tile_clang.exe
