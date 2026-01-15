#!/usr/bin/env bash
set -e

PYTHON_VERSION=3.10
PY_INC=$(python${PYTHON_VERSION} -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PY_LIB=$(python${PYTHON_VERSION} -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")

SRC_DIR=./tinytensor/engine
OUT_DIR=$SRC_DIR

# .c compile
gcc -O3 -fPIC -shared \
    -I"$PY_INC" \
    -I"$SRC_DIR" \
    "$SRC_DIR/cpu.c" \
    "$SRC_DIR/tensor.c" \
    -o "$OUT_DIR/cpu.so"

# .cu compile
nvcc -gencode arch=compute_75,code=sm_75 -O3 -Xcompiler -fPIC -shared \
    -I"$PY_INC" \
    -I"$SRC_DIR" \
    "$SRC_DIR/gpu_cuda.cu" \
    "$SRC_DIR/tensor.c" \
    -o "$OUT_DIR/gpu_cuda.so"
echo "Build successful"
