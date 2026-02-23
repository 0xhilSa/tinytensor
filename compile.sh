#!/usr/bin/env bash
set -e

spinner() {
  local pid=$1
  local spin='|/-\'
  local i=0
  while kill -0 "$pid" 2>/dev/null; do
    i=$(( (i+1) %4 ))
    printf "\r[%c] Compiling..." "${spin:$i:1}"
    sleep 0.1
  done
}

run_with_spinner() {
  "$@" &
  local pid=$!
  spinner $pid
  wait $pid
  printf "\r[✓] Done           \n"
}

PYTHON_VERSION=3.10
PY_INC=$(python${PYTHON_VERSION} -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PY_LIB=$(python${PYTHON_VERSION} -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")

TEN_SRC=./tinytensor/engine
C_SRC_DIR=./tinytensor/engine/cpu
C_OUT_DIR=$C_SRC_DIR
CU_SRC_DIR=./tinytensor/engine/cuda
CU_OUT_DIR=$CU_SRC_DIR

# compile ./tinytensor/engine/cpu/cpu.c
echo "compiling $C_SRC_DIR/cpu.c -> $C_SRC_DIR/cpu.so"
run_with_spinner nvcc -gencode arch=compute_86,code=sm_86 -O3 -Xcompiler -fPIC -shared \
  -I"$PY_INC" \
  -I"$C_SRC_DIR" \
  "$C_SRC_DIR/cpu.c" \
  "$TEN_SRC/tensor.c" \
  -lcudart \
  -o "$C_OUT_DIR/cpu.so"

# compile ./tinytensor/engine/constants.c
echo "compiling $TEN_SRC/constants.c -> $TEN_SRC/constants.so"
run_with_spinner nvcc -gencode arch=compute_86,code=sm_86 -O3 -Xcompiler -fPIC -shared \
  -I"$PY_INC" \
  "$TEN_SRC/constants.c" \
  -o "$TEN_SRC/constants.so"

# compile ./tinytensor/engine/cpu/cpu_ops.c
echo "compiling $C_SRC_DIR/cpu_ops.c -> $C_SRC_DIR/cpu_ops.so"
run_with_spinner nvcc -gencode arch=compute_86,code=sm_86 -O3 -Xcompiler -fPIC -shared \
  -I"$PY_INC" \
  -I"$C_SRC_DIR" \
  "$C_SRC_DIR/cpu_ops.c" \
  "$TEN_SRC/tensor.c" \
  -lcudart \
  -o "$C_OUT_DIR/cpu_ops.so"

# compile ./tinytensor/engine/cuda/cuda.cu
echo "compiling $CU_SRC_DIR/cuda.cu -> $CU_SRC_DIR/cuda.so"
run_with_spinner nvcc -gencode arch=compute_86,code=sm_86 -O3 -Xcompiler -fPIC -shared \
  -I"$PY_INC" \
  -I"$CU_SRC_DIR" \
  "$CU_SRC_DIR/cuda.cu" \
  "$TEN_SRC/tensor.c" \
  -lnvidia-ml \
  -o "$CU_OUT_DIR/cuda.so"

# compile ./tinytensor/engine/cuda/cuda_ops.cu
echo "compiling $CU_SRC_DIR/cuda_ops.cu -> $CU_SRC_DIR/cuda_ops.so"
run_with_spinner nvcc -gencode arch=compute_86,code=sm_86 -O3 -Xcompiler -fPIC -shared \
  -I"$PY_INC" \
  -I"$CU_SRC_DIR" \
  "$CU_SRC_DIR/cuda_ops.cu" \
  "$TEN_SRC/tensor.c" \
  -o "$CU_OUT_DIR/cuda_ops.so"

# compile ./tinytensor/engine/cuda/functional_cuda.cu
echo "compiling $CU_SRC_DIR/functional_cuda.cu -> $CU_SRC_DIR/functional_cuda.so"
run_with_spinner nvcc -gencode arch=compute_86,code=sm_86 -O3 -Xcompiler -fPIC -shared \
  -I"$PY_INC" \
  -I"$CU_SRC_DIR" \
  "$CU_SRC_DIR/functional_cuda.cu" \
  "$TEN_SRC/tensor.c" \
  -o "$CU_OUT_DIR/functional_cuda.so"

# compile ./tinytensor/engine/cpu/functional_cpu.c
echo "compiling $C_SRC_DIR/functional_cpu.cu -> $C_SRC_DIR/functional_cpu.so"
run_with_spinner nvcc -gencode arch=compute_86,code=sm_86 -O3 -Xcompiler -fPIC -shared \
  -I"$PY_INC" \
  -I"$C_SRC_DIR" \
  "$C_SRC_DIR/functional_cpu.c" \
  "$TEN_SRC/tensor.c" \
  -o "$C_OUT_DIR/functional_cpu.so"


# versions
VERSION="0.3.0"
GIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "unknow")
CUDA_VERSION=$(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')
VERSION="${VERSION}+cu${CUDA_VERSION//./}"

cat > tinytensor/version.py << EOF
# auto generted
from typing import Optional

__all__ = ["__version__", "cuda", "git_version"]
__version__ = "${VERSION}"
cuda: Optional[str] = "${CUDA_VERSION}"
git_version = "${GIT_HASH}"
EOF

pip install -r ./requirements.txt
pip install -e .

if [ "$1" == "-run" ]; then
  pytest -v
fi
