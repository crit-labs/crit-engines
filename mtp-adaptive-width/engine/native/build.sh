#!/bin/bash
# Builds libq38sampling.so (crit Inference token-sampling + C ABI) for sm_120.
set -e
cd "$(dirname "$0")"
NVCC=${NVCC:-/usr/local/cuda-13.1/bin/nvcc}
$NVCC -O3 -std=c++20 -shared -Xcompiler -fPIC -cudart shared -gencode arch=compute_120,code=sm_120 \
  -Iinclude -I. sampling.cu sampling_capi.cu -o libq38sampling.so
