@echo off
cd /d D:\Lumina-Engine\Lumina-Engine-Bin
set GGML_Q1_LUT=
set GGML_Q1_LUT_DEBUG=1
Lumina-Engine-cli.exe -m ..\Bonsai-8B.gguf -p "The capital of France is" -n 24 --temp 0 --seed 42 -t 6 -st > on_dbg.txt 2>&1
