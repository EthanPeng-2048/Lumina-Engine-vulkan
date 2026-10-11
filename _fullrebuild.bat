@echo off
rd /s /q "D:\Lumina-Engine\llama.cpp-b10991\build-release"
call "D:\Lumina-Engine\configure-release.bat"
if errorlevel 1 ( echo CONFIG_FAILED & exit /b 1 )
call "D:\Lumina-Engine\bench\build_all_msvc.bat"
if errorlevel 1 ( echo BUILD_FAILED & exit /b 1 )
call "D:\Lumina-Engine\bench\assemble_bin.bat"
echo FULLREBUILD_EXIT=%ERRORLEVEL% at %DATE% %TIME%
