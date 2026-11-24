@echo off
rem Builds the engine's CUDA kernels for this GPU under MSVC's environment with CUDA 13.4's nvcc and the run's Python
rem with torch: gguf-projections' code/gguf_gemv.cu (GGUF GEMVs, int8 prompt MMQ, expert glue, routing),
rem qwen35-attention's code/attention.cu and qwen35-recurrence's code/gdn.cu. Output: each plugin's code\build\*.pyd.
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
if not defined CUDA_HOME set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
rem python below is the interpreter that ran build.py (crit's, with torch): build.py puts its folder first on PATH
set "PATH=%CUDA_HOME%\bin;%PATH%"
for %%P in (gguf-projections qwen35-attention qwen35-recurrence qwen35-norms gguf-experts) do (
  if exist "%~dp0plugins\%%P\code\build_kernels.py" python "%~dp0plugins\%%P\code\build_kernels.py" || exit /b 1
)
