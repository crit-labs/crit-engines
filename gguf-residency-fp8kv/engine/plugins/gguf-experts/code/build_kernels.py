"""Build expert_tier (code/expert_tier.cu) for this machine's GPU into code/build: run under MSVC's environment (vcvars64)
with nvcc on PATH. The plugin loads code/build/expert_tier*.pyd (or .so) where it exists and otherwise decodes in torch."""
import os
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent
out = HERE / "build"
out.mkdir(exist_ok=True)
major, minor = torch.cuda.get_device_capability()
arch = f"{major}{minor}"
module = load(name="expert_tier", sources=[str(HERE / "expert_tier.cu")], build_directory=str(out),
              extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-Xcompiler", "/Zc:preprocessor", "-Xcompiler", "/arch:AVX2", "-Xcompiler", "/O2"],
              extra_cflags=["/Zc:preprocessor", "/arch:AVX2", "/O2"] if os.name == "nt" else [],
              extra_include_paths=[str(HERE)], verbose=True)
print("built", module.__file__)
sys.exit(0)
