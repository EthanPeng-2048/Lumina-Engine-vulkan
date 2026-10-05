@echo off
cd /d D:\Lumina-Engine\Lumina-Engine-Bin
set GGML_Q1_LUT=0
Lumina-Engine-cli.exe -m ..\Bonsai-8B.gguf -p "The capital of France is" -n 24 --temp 0 --seed 42 -t 6 -st > corr_off.txt 2>&1
set GGML_Q1_LUT=
Lumina-Engine-cli.exe -m ..\Bonsai-8B.gguf -p "The capital of France is" -n 24 --temp 0 --seed 42 -t 6 -st > corr_on.txt 2>&1
fc corr_off.txt corr_on.txt
