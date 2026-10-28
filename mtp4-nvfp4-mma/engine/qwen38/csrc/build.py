"""Build csrc/nvfp4_mma.cu for this device (sm_120a) with torch.utils.cpp_extension: the package's build command."""
import os
import shutil
import sys
import tempfile

from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "qwen38_nvfp4_mma"


def main():
    build = tempfile.mkdtemp(prefix="nvfp4-build-")
    try:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
        load(name=NAME, sources=[os.path.join(HERE, "nvfp4_mma.cu")], extra_cuda_cflags=["-O3", "-lineinfo"],
             build_directory=build, verbose=False)
        shutil.copy(os.path.join(build, NAME + ".so"), os.path.join(HERE, NAME + ".so"))
    finally:
        shutil.rmtree(build, ignore_errors=True)
    print("built", os.path.join(HERE, NAME + ".so"))


if __name__ == "__main__":
    sys.exit(main())
