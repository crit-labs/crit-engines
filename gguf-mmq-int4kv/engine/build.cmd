@echo off
rem Builds the engine's CUDA kernels for this GPU: gguf-projections' code/gguf_gemv.cu (the GGUF GEMVs, expert glue and
rem routing), gemma4-attention's code/decode_attention.cu and gemma4-norms' code/norms.cu, under MSVC's environment
rem with CUDA 13.4's nvcc and the run's Python with torch. Output: each plugin's code\build\*.pyd.
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
if not defined CUDA_HOME set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
rem python below is the interpreter that ran build.py (crit's, with torch): build.py puts its folder first on PATH
set "PATH=%CUDA_HOME%\bin;%PATH%"
python "%~dp0plugins\gguf-projections\code\build_kernels.py" || exit /b 1
python "%~dp0plugins\gemma4-attention\code\build_kernels.py" || exit /b 1
python "%~dp0plugins\gemma4-norms\code\build_kernels.py" || exit /b 1
