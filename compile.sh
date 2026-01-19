#!/usr/bin/env bash
set -e

#hilsa:/mnt/d/fun/tinytensor/tinytensor/engine$ ls
#__init__.py  __pycache__  cpu  cuda  dtypes.h  tensor.c  tensor.h

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


# .c compile
echo "building $C_SRC_DIR/cpu.so"
run_with_spinner nvcc -gencode arch=compute_75,code=sm_75 -O3 -Xcompiler -fPIC -shared \
  -I"$PY_INC" \
  -I"$C_SRC_DIR" \
  "$C_SRC_DIR/cpu.c" \
  "$TEN_SRC/tensor.c" \
  -lcudart \
  -o "$C_OUT_DIR/cpu.so"

# .cu compile
echo "building $CU_SRC_DIR/gpu_cuda.so"
run_with_spinner nvcc -gencode arch=compute_75,code=sm_75 -O3 -Xcompiler -fPIC -shared \
  -I"$PY_INC" \
  -I"$CU_SRC_DIR" \
  "$CU_SRC_DIR/gpu_cuda.cu" \
  "$TEN_SRC/tensor.c" \
  -o "$CU_OUT_DIR/gpu_cuda.so"
