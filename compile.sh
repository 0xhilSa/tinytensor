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

TEN_SRC=./tinytensor/engine/
C_SRC_DIR=./tinytensor/engine/cpu
C_OUT_DIR=$C_SRC_DIR
CU_SRC_DIR=./tinytensor/engine/cuda
CU_OUT_DIR=$CU_SRC_DIR

# compile ./tinytensor/engine/cpu/cpu.c
echo "compiling $C_SRC_DIR/cpu.c -> $C_SRC_DIR/cpu.so"
run_with_spinner nvcc -gencode arch=compute_75,code=sm_75 -O3 -Xcompiler -fPIC -shared \
  -I"$PY_INC" \
  -I"$C_SRC_DIR" \
  "$C_SRC_DIR/cpu.c" \
  "$TEN_SRC/tensor.c" \
  -lcudart \
  -o "$C_OUT_DIR/cpu.so"

# compile ./tinytensor/engine/cpu/cpu_ops.c
echo "compiling $C_SRC_DIR/cpu_ops.c -> $C_SRC_DIR/cpu_ops.so"
run_with_spinner nvcc -gencode arch=compute_75,code=sm_75 -O3 -Xcompiler -fPIC -shared \
  -I"$PY_INC" \
  -I"$C_SRC_DIR" \
  "$C_SRC_DIR/cpu_ops.c" \
  "$TEN_SRC/tensor.c" \
  -lcudart \
  -o "$C_OUT_DIR/cpu_ops.so"

# compile ./tinytensor/engine/cuda/cuda.cu
echo "compiling $CU_SRC_DIR/cuda.cu -> $CU_SRC_DIR/cuda.so"
run_with_spinner nvcc -gencode arch=compute_75,code=sm_75 -O3 -Xcompiler -fPIC -shared \
  -I"$PY_INC" \
  -I"$CU_SRC_DIR" \
  "$CU_SRC_DIR/cuda.cu" \
  "$TEN_SRC/tensor.c" \
  -lnvidia-ml \
  -o "$CU_OUT_DIR/cuda.so"

# compile ./tinytensor/engine/cuda/cuda_ops.cu
echo "compiling $CU_SRC_DIR/cuda_ops.cu -> $CU_SRC_DIR/cuda_ops.so"
run_with_spinner nvcc -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_86,code=compute_86 \
  -O3 -Xcompiler -fPIC -shared \
  -I"$PY_INC" \
  -I"$CU_SRC_DIR" \
  "$CU_SRC_DIR/cuda_ops.cu" \
  "$TEN_SRC/tensor.c" \
  -o "$CU_OUT_DIR/cuda_ops.so"
