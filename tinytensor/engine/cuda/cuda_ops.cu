#include <python3.10/Python.h>
#include <cuda_runtime.h>
#include "../tensor.h"

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      PyErr_Format(PyExc_RuntimeError, "CUDA error: %s", cudaGetErrorString(err)); \
      return NULL; \
    } \
  } while(0)

void capsule_destroyer(PyObject *capsule){
  tensor_t *t = (tensor_t *)PyCapsule_GetPointer(capsule, "tensor_t on CUDA");
  if(t){
    destroy(t);
  }
}

typedef enum {L, R} side_t;

// CUDA kernel for element-wise tensor addition
template<typename T>
__global__ void add_tensor_kernel(const T *x, const T *y, T *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = x[idx] + y[idx];
  }
}

// Specialized kernel for boolean OR operation
__global__ void add_bool_kernel(const bool *x, const bool *y, bool *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = x[idx] || y[idx];
  }
}

// CUDA kernel for complex64 addition
__global__ void add_complex64_kernel(const complex64 *x, const complex64 *y, complex64 *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx].real = x[idx].real + y[idx].real;
    z[idx].imag = x[idx].imag + y[idx].imag;
  }
}

// CUDA kernel for complex128 addition
__global__ void add_complex128_kernel(const complex128 *x, const complex128 *y, complex128 *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx].real = x[idx].real + y[idx].real;
    z[idx].imag = x[idx].imag + y[idx].imag;
  }
}

// CUDA kernel for scalar-tensor addition
template<typename T>
__global__ void add_scalar_tensor_kernel(const T *tensor, const T *scalar, T *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = tensor[idx] + (*scalar);
  }
}

// Specialized kernel for boolean OR with scalar
__global__ void add_scalar_bool_kernel(const bool *tensor, const bool *scalar, bool *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = tensor[idx] || (*scalar);
  }
}

// CUDA kernel for scalar-tensor complex64 addition
__global__ void add_scalar_complex64_kernel(const complex64 *tensor, const complex64 *scalar, complex64 *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx].real = tensor[idx].real + scalar->real;
    z[idx].imag = tensor[idx].imag + scalar->imag;
  }
}

// CUDA kernel for scalar-tensor complex128 addition
__global__ void add_scalar_complex128_kernel(const complex128 *tensor, const complex128 *scalar, complex128 *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx].real = tensor[idx].real + scalar->real;
    z[idx].imag = tensor[idx].imag + scalar->imag;
  }
}

static tensor_t *alloc_result_tensor(const tensor_t *t){
  tensor_t *tz = (tensor_t *)malloc(sizeof(tensor_t));
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
    return NULL;
  }
  tz->dtype = t->dtype;
  tz->size = t->size;
  tz->ndim = t->ndim;
  tz->element_size = t->element_size;
  tz->device = t->device;
  tz->stride = NULL;

  if(t->ndim > 0 && t->shape){
    tz->shape = (size_t *)malloc(sizeof(size_t) * tz->ndim);
    if(!tz->shape){
      free(tz);
      PyErr_SetString(PyExc_RuntimeError, "tensor_t.shape allocation failed!");
      return NULL;
    }
    for(size_t i = 0; i < tz->ndim; i++){
      tz->shape[i] = t->shape[i];
    }
  } else {
    tz->shape = NULL;
  }

  tz->storage = (storage_t *)malloc(sizeof(storage_t));
  if(!tz->storage){
    if(tz->shape) free(tz->shape);
    free(tz);
    PyErr_SetString(PyExc_RuntimeError, "tensor_t.storage allocation failed!");
    return NULL;
  }

  tz->storage->bytes = tz->size * tz->element_size;
  tz->storage->device = tz->device;
  tz->storage->refcount = 1;

  // Allocate GPU memory
  cudaError_t err = cudaMalloc(&tz->storage->ptr, tz->storage->bytes);
  if(err != cudaSuccess){
    if(tz->shape) free(tz->shape);
    free(tz->storage);
    free(tz);
    PyErr_Format(PyExc_RuntimeError, "CUDA malloc failed: %s", cudaGetErrorString(err));
    return NULL;
  }

  tz->buf = tz->storage->ptr;
  return tz;
}

static PyObject *__add_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;

  // Calculate grid and block dimensions
  int blockSize = 256;
  int gridSize = (length + blockSize - 1) / blockSize;

  // Launch appropriate kernel based on dtype
  switch(dtype){
    case BOOL:
      add_bool_kernel<<<gridSize, blockSize>>>((bool *)tx->buf, (bool *)ty->buf, (bool *)tz->buf, length);
      break;
    case INT8:
      add_tensor_kernel<int8><<<gridSize, blockSize>>>((int8 *)tx->buf, (int8 *)ty->buf, (int8 *)tz->buf, length);
      break;
    case UINT8:
      add_tensor_kernel<uint8><<<gridSize, blockSize>>>((uint8 *)tx->buf, (uint8 *)ty->buf, (uint8 *)tz->buf, length);
      break;
    case INT16:
      add_tensor_kernel<int16><<<gridSize, blockSize>>>((int16 *)tx->buf, (int16 *)ty->buf, (int16 *)tz->buf, length);
      break;
    case UINT16:
      add_tensor_kernel<uint16><<<gridSize, blockSize>>>((uint16 *)tx->buf, (uint16 *)ty->buf, (uint16 *)tz->buf, length);
      break;
    case INT32:
      add_tensor_kernel<int32><<<gridSize, blockSize>>>((int32 *)tx->buf, (int32 *)ty->buf, (int32 *)tz->buf, length);
      break;
    case UINT32:
      add_tensor_kernel<uint32><<<gridSize, blockSize>>>((uint32 *)tx->buf, (uint32 *)ty->buf, (uint32 *)tz->buf, length);
      break;
    case INT64:
      add_tensor_kernel<int64><<<gridSize, blockSize>>>((int64 *)tx->buf, (int64 *)ty->buf, (int64 *)tz->buf, length);
      break;
    case UINT64:
      add_tensor_kernel<uint64><<<gridSize, blockSize>>>((uint64 *)tx->buf, (uint64 *)ty->buf, (uint64 *)tz->buf, length);
      break;
    case FP32:
      add_tensor_kernel<float32><<<gridSize, blockSize>>>((float32 *)tx->buf, (float32 *)ty->buf, (float32 *)tz->buf, length);
      break;
    case FP64:
      add_tensor_kernel<float64><<<gridSize, blockSize>>>((float64 *)tx->buf, (float64 *)ty->buf, (float64 *)tz->buf, length);
      break;
    case FP128:
      add_tensor_kernel<float128><<<gridSize, blockSize>>>((float128 *)tx->buf, (float128 *)ty->buf, (float128 *)tz->buf, length);
      break;
    case CMPX64:
      add_complex64_kernel<<<gridSize, blockSize>>>((complex64 *)tx->buf, (complex64 *)ty->buf, (complex64 *)tz->buf, length);
      break;
    case CMPX128:
      add_complex128_kernel<<<gridSize, blockSize>>>((complex128 *)tx->buf, (complex128 *)ty->buf, (complex128 *)tz->buf, length);
      break;
    case ERROR:
      PyErr_SetString(PyExc_RuntimeError, "something is wrong i can feel it");
      return NULL;
  }

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA kernel launch failed: %s", cudaGetErrorString(err));
    return NULL;
  }

  // Wait for kernel to complete
  err = cudaDeviceSynchronize();
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA synchronization failed: %s", cudaGetErrorString(err));
    return NULL;
  }

  return PyCapsule_New(tz, "tensor_t on CUDA", capsule_destroyer);
}

static PyObject *__add_scalar_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz, side_t side){
  const tensor_t *tensor = (side == L) ? ty : tx;
  const tensor_t *scalar = (side == L) ? tx : ty;
  dtype_t dtype = tensor->dtype;
  size_t length = tensor->size;

  // Calculate grid and block dimensions
  int blockSize = 256;
  int gridSize = (length + blockSize - 1) / blockSize;

  // Launch appropriate kernel based on dtype
  switch(dtype){
    case BOOL:
      add_scalar_bool_kernel<<<gridSize, blockSize>>>((bool *)tensor->buf, (bool *)scalar->buf, (bool *)tz->buf, length);
      break;
    case INT8:
      add_scalar_tensor_kernel<int8><<<gridSize, blockSize>>>((int8 *)tensor->buf, (int8 *)scalar->buf, (int8 *)tz->buf, length);
      break;
    case UINT8:
      add_scalar_tensor_kernel<uint8><<<gridSize, blockSize>>>((uint8 *)tensor->buf, (uint8 *)scalar->buf, (uint8 *)tz->buf, length);
      break;
    case INT16:
      add_scalar_tensor_kernel<int16><<<gridSize, blockSize>>>((int16 *)tensor->buf, (int16 *)scalar->buf, (int16 *)tz->buf, length);
      break;
    case UINT16:
      add_scalar_tensor_kernel<uint16><<<gridSize, blockSize>>>((uint16 *)tensor->buf, (uint16 *)scalar->buf, (uint16 *)tz->buf, length);
      break;
    case INT32:
      add_scalar_tensor_kernel<int32><<<gridSize, blockSize>>>((int32 *)tensor->buf, (int32 *)scalar->buf, (int32 *)tz->buf, length);
      break;
    case UINT32:
      add_scalar_tensor_kernel<uint32><<<gridSize, blockSize>>>((uint32 *)tensor->buf, (uint32 *)scalar->buf, (uint32 *)tz->buf, length);
      break;
    case INT64:
      add_scalar_tensor_kernel<int64><<<gridSize, blockSize>>>((int64 *)tensor->buf, (int64 *)scalar->buf, (int64 *)tz->buf, length);
      break;
    case UINT64:
      add_scalar_tensor_kernel<uint64><<<gridSize, blockSize>>>((uint64 *)tensor->buf, (uint64 *)scalar->buf, (uint64 *)tz->buf, length);
      break;
    case FP32:
      add_scalar_tensor_kernel<float32><<<gridSize, blockSize>>>((float32 *)tensor->buf, (float32 *)scalar->buf, (float32 *)tz->buf, length);
      break;
    case FP64:
      add_scalar_tensor_kernel<float64><<<gridSize, blockSize>>>((float64 *)tensor->buf, (float64 *)scalar->buf, (float64 *)tz->buf, length);
      break;
    case FP128:
      add_scalar_tensor_kernel<float128><<<gridSize, blockSize>>>((float128 *)tensor->buf, (float128 *)scalar->buf, (float128 *)tz->buf, length);
      break;
    case CMPX64:
      add_scalar_complex64_kernel<<<gridSize, blockSize>>>((complex64 *)tensor->buf, (complex64 *)scalar->buf, (complex64 *)tz->buf, length);
      break;
    case CMPX128:
      add_scalar_complex128_kernel<<<gridSize, blockSize>>>((complex128 *)tensor->buf, (complex128 *)scalar->buf, (complex128 *)tz->buf, length);
      break;
    case ERROR:
      PyErr_SetString(PyExc_RuntimeError, "something is wrong i can feel it");
      return NULL;
  }

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA kernel launch failed: %s", cudaGetErrorString(err));
    return NULL;
  }

  // Wait for kernel to complete
  err = cudaDeviceSynchronize();
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA synchronization failed: %s", cudaGetErrorString(err));
    return NULL;
  }

  return PyCapsule_New(tz, "tensor_t on CUDA", capsule_destroyer);
}

static int shapes_match(const tensor_t *tx, const tensor_t *ty){
  if(tx->ndim != ty->ndim) return 0;
  if(!tx->shape || !ty->shape) return 0;
  for(size_t i = 0; i < tx->ndim; i++){
    if(tx->shape[i] != ty->shape[i]) return 0;
  }
  return 1;
}

static PyObject *add(PyObject *self, PyObject *args){
  PyObject *x, *y;
  if(!PyArg_ParseTuple(args, "OO", &x, &y)) return NULL;

  if(!PyCapsule_CheckExact(x) || !PyCapsule_CheckExact(y)){
    PyErr_SetString(PyExc_RuntimeError, "operands must be the tensor capsule");
    return NULL;
  }

  tensor_t *tx = (tensor_t *)PyCapsule_GetPointer(x, "tensor_t on CUDA");
  tensor_t *ty = (tensor_t *)PyCapsule_GetPointer(y, "tensor_t on CUDA");

  if(!tx || !ty){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }

  if(tx->dtype != ty->dtype){
    PyErr_SetString(PyExc_TypeError, "Both tensor_t(s) must have the same dtype");
    return NULL;
  }

  if(tx->device.type != ty->device.type || tx->device.type != CUDA){
    PyErr_SetString(PyExc_RuntimeError, "Both tensor_t(s) must be on same device (CUDA)");
    return NULL;
  }

  int x_is_scalar = (tx->size == 1 && tx->ndim == 0);
  int y_is_scalar = (ty->size == 1 && ty->ndim == 0);
  tensor_t *tz = NULL;

  if(!x_is_scalar && !y_is_scalar){
    if(!shapes_match(tx, ty)){
      PyErr_SetString(PyExc_ValueError, "Tensor shapes do not match for element-wise addition");
      return NULL;
    }

    tz = alloc_result_tensor(tx);
    if(!tz) return NULL;
    return __add_tensor__(tx, ty, tz);
  } else if(x_is_scalar && !y_is_scalar){
    tz = alloc_result_tensor(ty);
    if(!tz) return NULL;
    return __add_scalar_tensor__(tx, ty, tz, L);
  } else if(!x_is_scalar && y_is_scalar){
    tz = alloc_result_tensor(tx);
    if(!tz) return NULL;
    return __add_scalar_tensor__(tx, ty, tz, R);
  } else {
    tz = alloc_result_tensor(tx);
    if(!tz) return NULL;
    return __add_tensor__(tx, ty, tz);
  }
}

static PyMethodDef methods[] = {
  {"add", add, METH_VARARGS, "element-wise addition of two tensors or tensor with scalar on CUDA"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "cuda_ops",
  "CUDA tensor operations module",
  -1,
  methods
};

PyMODINIT_FUNC PyInit_cuda_ops(void){
  return PyModule_Create(&module);
}
