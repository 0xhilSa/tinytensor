#include <cuda_runtime.h>
#include <cuComplex.h>
#include <python3.10/Python.h>
#include "../tensor.h"

#define KERNEL_LAUNCH(kernel, blocks, threads, ...)  \
  do{ \
    kernel<<<blocks, threads>>>(__VA_ARGS__); \
  } while (0)

typedef enum {
  SCALAR_RIGHT,
  SCALAR_LEFT
} scalar_side_t;

static void capsule_destructor(PyObject *capsule){
  tensor_t *t = (tensor_t *)PyCapsule_GetPointer(capsule, "tensor_t on CUDA");
  if(!t) return;
  cudaFree(t->data);
  destroy(t);
  free(t);
}

__global__ void add_tensor_bool(const bool *x, const bool *y, bool *out, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length) out[idx] = x[idx] | y[idx];
}

__global__ void add_tensor_int8(const i8 *x, const i8 *y, i8 *out, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length) out[idx] = x[idx] + y[idx];
}

__global__ void add_tensor_uint8(const u8 *x, const u8 *y, u8 *out, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length) out[idx] = x[idx] + y[idx];
}

__global__ void add_tensor_int16(const i16 *x, const i16 *y, i16 *out, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length) out[idx] = x[idx] + y[idx];
}

__global__ void add_tensor_uint16(const u16 *x, const u16 *y, u16 *out, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length) out[idx] = x[idx] + y[idx];
}

__global__ void add_tensor_int32(const i32 *x, const i32 *y, i32 *out, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length) out[idx] = x[idx] + y[idx];
}

__global__ void add_tensor_uint32(const u32 *x, const u32 *y, u32 *out, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length) out[idx] = x[idx] + y[idx];
}

__global__ void add_tensor_int64(const i64 *x, const i64 *y, i64 *out, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length) out[idx] = x[idx] + y[idx];
}

__global__ void add_tensor_uint64(const u64 *x, const u64 *y, u64 *out, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length) out[idx] = x[idx] + y[idx];
}

__global__ void add_tensor_float32(const f32 *x, const f32 *y, f32 *out, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length) out[idx] = x[idx] + y[idx];
}

__global__ void add_tensor_float64(const f64 *x, const f64 *y, f64 *out, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length) out[idx] = x[idx] + y[idx];
}

__global__ void add_tensor_float128(const f128 *x, const f128 *y, f128 *out, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length) out[idx] = x[idx] + y[idx];
}

__global__ void add_tensor_complex64(const cuFloatComplex *x, const cuFloatComplex *y, cuFloatComplex *out, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    out[idx].x = x[idx].x + y[idx].x;
    out[idx].y = x[idx].y + y[idx].y;
  }
}

__global__ void add_tensor_complex128(const cuDoubleComplex *x, const cuDoubleComplex *y, cuDoubleComplex *out, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    out[idx].x = x[idx].x + y[idx].x;
    out[idx].y = x[idx].y + y[idx].y;
  }
}

__global__ void add_tensor_scalar_bool(const bool *x, const bool *y, bool *out, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length) out[idx] = x[idx] | *y;
}
__global__ void add_tensor_scalar_int8(const i8 *x, const i8 *y, i8 *out, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length) out[idx] = x[idx] + *y;
}

__global__ void add_tensor_scalar_uint8(const u8 *x, const u8 *y, u8 *out, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length) out[idx] = x[idx] + *y;
}

__global__ void add_tensor_scalar_int16(const i16 *x, const i16 *y, i16 *out, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length) out[idx] = x[idx] + *y;
}

__global__ void add_tensor_scalar_uint16(const u16 *x, const u16 *y, u16 *out, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length) out[idx] = x[idx] + *y;
}

__global__ void add_tensor_scalar_int32(const i32 *x, const i32 *y, i32 *out, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length) out[idx] = x[idx] + *y;
}

__global__ void add_tensor_scalar_uint32(const u32 *x, const u32 *y, u32 *out, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length) out[idx] = x[idx] + *y;
}

__global__ void add_tensor_scalar_int64(const i64 *x, const i64 *y, i64 *out, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length) out[idx] = *y + x[idx];
}

__global__ void add_tensor_scalar_uint64(const u64 *x, const u64 *y, u64 *out, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length) out[idx] = x[idx] + *y;
}

__global__ void add_tensor_scalar_float32(const f32 *x, const f32 *y, f32 *out, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length) out[idx] = x[idx] + *y;
}

__global__ void add_tensor_scalar_float64(const f64 *x, const f64 *y, f64 *out, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length) out[idx] = x[idx] + *y;
}

__global__ void add_tensor_scalar_float128(const f128 *x, const f128 *y, f128 *out, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length) out[idx] = x[idx] + *y;
}

__global__ void add_tensor_scalar_complex64(const cuFloatComplex *x, const cuFloatComplex *y, cuFloatComplex *out, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
      out[idx].x = y->x + x[idx].x;
      out[idx].y = y->y + x[idx].y;
  }
}

__global__ void add_tensor_scalar_complex128(const cuDoubleComplex *x, const cuDoubleComplex *y, cuDoubleComplex *out, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
      out[idx].x = y->x + x[idx].x;
      out[idx].y = y->y + x[idx].y;
  }
}

void __add_tensor_kernel_dispatch__(const void *x, const void *y, void *out, size_t length, dtype_t dtype){
  int threads = 512;
  int blocks = (length + threads - 1) / threads;
  switch(dtype){
    case BOOL: KERNEL_LAUNCH(add_tensor_bool, blocks, threads, (bool *)x, (bool *)y, (bool *)out, length); break;
    case INT8: KERNEL_LAUNCH(add_tensor_int8, blocks, threads, (i8 *)x, (i8 *)y, (i8 *)out, length); break;
    case UINT8: KERNEL_LAUNCH(add_tensor_uint8, blocks, threads, (u8 *)x, (u8 *)y, (u8 *)out, length); break;
    case INT16: KERNEL_LAUNCH(add_tensor_int16, blocks, threads, (i16 *)x, (i16 *)y, (i16 *)out, length); break;
    case UINT16: KERNEL_LAUNCH(add_tensor_uint16, blocks, threads, (u16 *)x, (u16 *)y, (u16 *)out, length); break;
    case INT32: KERNEL_LAUNCH(add_tensor_int32, blocks, threads, (i32 *)x, (i32 *)y, (i32 *)out, length); break;
    case UINT32: KERNEL_LAUNCH(add_tensor_uint32, blocks, threads, (u32 *)x, (u32 *)y, (u32 *)out, length); break;
    case INT64: KERNEL_LAUNCH(add_tensor_int64, blocks, threads, (i64 *)x, (i64 *)y, (i64 *)out, length); break;
    case UINT64: KERNEL_LAUNCH(add_tensor_uint64, blocks, threads, (u64 *)x, (u64 *)y, (u64 *)out, length); break;
    case FP32: KERNEL_LAUNCH(add_tensor_float32, blocks, threads, (f32 *)x, (f32 *)y, (f32 *)out, length); break;
    case FP64: KERNEL_LAUNCH(add_tensor_float64, blocks, threads, (f64 *)x, (f64 *)y, (f64 *)out, length); break;
    case FP128: KERNEL_LAUNCH(add_tensor_float128, blocks, threads, (f128 *)x, (f128 *)y, (f128 *)out, length); break;
    case CMPX64: KERNEL_LAUNCH(add_tensor_complex64, blocks, threads, (cuFloatComplex *)x, (cuFloatComplex *)y, (cuFloatComplex *)out, length); break;
    case CMPX128: KERNEL_LAUNCH(add_tensor_complex128, blocks, threads, (cuDoubleComplex *)x, (cuDoubleComplex *)y, (cuDoubleComplex *)out, length); break;
    default: break;
  }
}

void __add_tensor_scalar_kernel_dispatch__(const void *x, const void *y, void *out, size_t length, dtype_t dtype){
  int threads = 512;
  int blocks = (length + threads - 1) / threads;
  switch(dtype){
    case BOOL: KERNEL_LAUNCH(add_tensor_scalar_bool, blocks, threads, (bool *)x, (bool *)y, (bool *)out, length); break;
    case INT8: KERNEL_LAUNCH(add_tensor_scalar_int8, blocks, threads, (i8 *)x, (i8 *)y, (i8 *)out, length); break;
    case UINT8: KERNEL_LAUNCH(add_tensor_scalar_uint8, blocks, threads, (u8 *)x, (u8 *)y, (u8 *)out, length); break;
    case INT16: KERNEL_LAUNCH(add_tensor_scalar_int16, blocks, threads, (i16 *)x, (i16 *)y, (i16 *)out, length); break;
    case UINT16: KERNEL_LAUNCH(add_tensor_scalar_uint16, blocks, threads, (u16 *)x, (u16 *)y, (u16 *)out, length); break;
    case INT32: KERNEL_LAUNCH(add_tensor_scalar_int32, blocks, threads, (i32 *)x, (i32 *)y, (i32 *)out, length); break;
    case UINT32: KERNEL_LAUNCH(add_tensor_scalar_uint32, blocks, threads, (u32 *)x, (u32 *)y, (u32 *)out, length); break;
    case INT64: KERNEL_LAUNCH(add_tensor_scalar_int64, blocks, threads, (i64 *)x, (i64 *)y, (i64 *)out, length); break;
    case UINT64: KERNEL_LAUNCH(add_tensor_scalar_uint64, blocks, threads, (u64 *)x, (u64 *)y, (u64 *)out, length); break;
    case FP32: KERNEL_LAUNCH(add_tensor_scalar_float32, blocks, threads, (f32 *)x, (f32 *)y, (f32 *)out, length); break;
    case FP64: KERNEL_LAUNCH(add_tensor_scalar_float64, blocks, threads, (f64 *)x, (f64 *)y, (f64 *)out, length); break;
    case FP128: KERNEL_LAUNCH(add_tensor_scalar_float128, blocks, threads, (f128 *)x, (f128 *)y, (f128 *)out, length); break;
    case CMPX64: KERNEL_LAUNCH(add_tensor_scalar_complex64, blocks, threads, (cuFloatComplex *)x, (cuFloatComplex *)y, (cuFloatComplex *)out, length); break;
    case CMPX128: KERNEL_LAUNCH(add_tensor_scalar_complex128, blocks, threads, (cuDoubleComplex *)x, (cuDoubleComplex *)y, (cuDoubleComplex *)out, length); break;
    default: break;
  }
}

static PyObject *add(PyObject *self, PyObject *args){
  PyObject *x, *y;
  if(!PyArg_ParseTuple(args, "OO", &x, &y)) return NULL;
  if(!PyCapsule_CheckExact(x) || !PyCapsule_CheckExact(y)){
    PyErr_SetString(PyExc_RuntimeError, "Both operands must be tensor capsules");
    return NULL;
  }
  tensor_t *tx = (tensor_t *)PyCapsule_GetPointer(x, "tensor_t on CUDA");
  tensor_t *ty = (tensor_t *)PyCapsule_GetPointer(y, "tensor_t on CUDA");
  if(!tx || !ty){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  if(tx->device != ty->device){
    PyErr_SetString(PyExc_RuntimeError, "Tensor device mismatch");
    return NULL;
  }
  if(tx->dtype != ty->dtype){
    PyErr_SetString(PyExc_RuntimeError, "Tensor dtype mismatch");
    return NULL;
  }
  if(tx->device_index != ty->device_index){
    PyErr_SetString(PyExc_RuntimeError, "Tensor device index mismatch");
    return NULL;
  }
  size_t bytes;
  tensor_t *tensor = NULL;
  tensor_t *scalar = NULL;
  if(tx->length == ty->length){
    tensor = tx;
    scalar = NULL;
    bytes = tx->length * tx->elem_size;
  }else if(tx->length == 1){
    tensor = ty;
    scalar = tx;
    bytes = ty->length * ty->elem_size;
  }else if(ty->length == 1){
    tensor = tx;
    scalar = ty;
    bytes = tx->length * tx->elem_size;
  }else{
    PyErr_SetString(PyExc_RuntimeError, "Tensor shape mismatch");
    return NULL;
  }
  if(tx->dtype != ty->dtype){
    PyErr_SetString(PyExc_RuntimeError, "Tensor dtype mismatch");
    return NULL;
  }
  void *dptr;
  cudaError_t err = cudaMalloc(&dptr, bytes);
  if(err != cudaSuccess){
    PyErr_SetString(PyExc_RuntimeError, cudaGetErrorString(err));
    cudaFree(dptr);
    return NULL;
  }
  tensor_t *out = (tensor_t *)malloc(sizeof(tensor_t));
  if(!out){
    PyErr_NoMemory();
    cudaFree(out);
    if(out){
      destroy(out);
      free(out);
    }
    return NULL;
  }
  *out = create(tensor->ndim, tensor->shape, CUDA, tx->device_index, tensor->dtype);
  if(scalar) __add_tensor_scalar_kernel_dispatch__(tensor->data, scalar->data, dptr, tensor->length, tensor->dtype);
  else __add_tensor_kernel_dispatch__(tx->data, ty->data, dptr, tx->length, tx->dtype);
  out->data = dptr;
  return PyCapsule_New(out, "tensor_t on CUDA", capsule_destructor);
}


static PyMethodDef methods[] = {
  {"add", add, METH_VARARGS, "add 2 tensors"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "cuda_ops",
  NULL,
  -1,
  methods
};

PyMODINIT_FUNC PyInit_cuda_ops(void){
  return PyModule_Create(&module);
}
