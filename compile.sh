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

SRC_DIR=./tinytensor/engine
OUT_DIR=$SRC_DIR


# .c compile
echo "building cpu.so"
run_with_spinner gcc -O3 -fPIC -shared \
    -I"$PY_INC" \
    -I"$SRC_DIR" \
    "$SRC_DIR/cpu.c" \
    "$SRC_DIR/tensor.c" \
    -o "$OUT_DIR/cpu.so"

# .cu compile
echo "building gpu_cuda.so"
run_with_spinner nvcc -gencode arch=compute_75,code=sm_75 -O3 -Xcompiler -fPIC -shared \
    -I"$PY_INC" \
    -I"$SRC_DIR" \
    "$SRC_DIR/gpu_cuda.cu" \
    "$SRC_DIR/tensor.c" \
    -o "$OUT_DIR/gpu_cuda.so"
