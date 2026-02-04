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

// CUDA kernel for element-wise tensor addition
template<typename T>
__global__ void add_tensor_kernel(const T *x, const T *y, T *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = x[idx] + y[idx];
  }
}

// CUDA kernel for element-wise tensor subtraction
template<typename T>
__global__ void sub_tensor_kernel(const T *x, const T *y, T *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = x[idx] - y[idx];
  }
}

// CUDA kernel for element-wise tensor multiplication
template<typename T>
__global__ void mul_tensor_kernel(const T *x, const T *y, T *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = x[idx] * y[idx];
  }
}

// CUDA kernel for element-wise tensor equivalence
template<typename T>
__global__ void eq_tensor_kernel(const T *x, const T *y, bool *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = (bool)(x[idx] == y[idx]);
  }
}

// CUDA kernel for element-wise cmpx tensor equivalence
template<typename T>
__global__ void eq_cmpx_tensor_kernel(const T *x, const T *y, bool *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = (bool)(x[idx].real == y[idx].real && x[idx].imag == y[idx].imag);
  }
}

// CUDA kernel for element-wise tensor non-equivalence
template<typename T>
__global__ void ne_tensor_kernel(const T *x, const T *y, bool *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = (bool)(x[idx] != y[idx]);
  }
}

// CUDA kernel for element-wise cmpx tensor non-equivalence
template<typename T>
__global__ void ne_cmpx_tensor_kernel(const T *x, const T *y, bool *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = (bool)(x[idx].real != y[idx].real || x[idx].imag != y[idx].imag);
  }
}

// CUDA kernel for element-wise tensor greater than
template<typename T>
__global__ void gt_tensor_kernel(const T *x, const T *y, bool *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = (x[idx] > y[idx]);
  }
}

// CUDA kernel for element-wise tensor greater than equal
template<typename T>
__global__ void ge_tensor_kernel(const T *x, const T *y, bool *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = (x[idx] > y[idx]);
  }
}

// CUDA kernel for element-wise tensor less than
template<typename T>
__global__ void lt_tensor_kernel(const T *x, const T *y, bool *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = (x[idx] < y[idx]);
  }
}

// CUDA kernel for element-wise tensor less than equal
template<typename T>
__global__ void le_tensor_kernel(const T *x, const T *y, bool *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = (x[idx] <= y[idx]);
  }
}

// CUDA kernel for element-wise tensor negations operation
template<typename T>
__global__ void neg_tensor_kernel(const T* x, T* z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = -x[idx];
  }
}

// CUDA kernel for element-wise tensor(cmpx) negations operation
template<typename T>
__global__ void neg_cmpx_tensor_kernel(const T* x, T* z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx].real = -x[idx].real;
    z[idx].imag = -x[idx].imag;
  }
}

// CUDA kernel for element-wise tensor non-negations operation
template<typename T>
__global__ void pos_tensor_kernel(const T* x, T* z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = x[idx];
  }
}

// CUDA kernel for element-wise tensor(cmpx) non-negations operation
template<typename T>
__global__ void pos_cmpx_tensor_kernel(const T* x, T* z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx].real = x[idx].real;
    z[idx].imag = x[idx].imag;
  }
}

// CUDA kernel for element-wise tensor abs operation
template<typename T>
__global__ void abs_tensor_kernel(const T* x, T*z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = (x[idx] < 0) ? -x[idx] : x[idx];
  }
}

// CUDA kernel for element-wise tensor(cmpx) abs operation
template<typename T, typename U>
__global__ void abs_cmpx_tensor_kernel(const T* x, U* z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = sqrt(x[idx].real * x[idx].real + x[idx].imag * x[idx].imag);
  }
}

// Specialized kernel for boolean OR operation
__global__ void add_bool_kernel(const bool *x, const bool *y, bool *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = x[idx] || y[idx];
  }
}

__global__ void sub_bool_kernel(const bool *x, const bool *y, bool *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = x[idx] - y[idx];
  }
}

// CUDA kernel for boolean AND op
__global__ void mul_bool_kernel(const bool *x, const bool *y, bool *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = x[idx] & y[idx];
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

// CUDA kernel for complex64 subtraction
__global__ void sub_complex64_kernel(const complex64 *x, const complex64 *y, complex64 *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx].real = x[idx].real - y[idx].real;
    z[idx].imag = x[idx].imag - y[idx].imag;
  }
}

// CUDA kernel for complex64 multiplication
__global__ void mul_complex64_kernel(const complex64 *x, const complex64 *y, complex64 *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx].real = x[idx].real * y[idx].real - x[idx].imag * y[idx].imag;
    z[idx].imag = x[idx].real * y[idx].imag - x[idx].imag * y[idx].real;
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

// CUDA kernel for complex128 subtraction
__global__ void sub_complex128_kernel(const complex128 *x, const complex128 *y, complex128 *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx].real = x[idx].real - y[idx].real;
    z[idx].imag = x[idx].imag - y[idx].imag;
  }
}

// CUDA kernel for complex64 multiplication
__global__ void mul_complex128_kernel(const complex128 *x, const complex128 *y, complex128 *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx].real = x[idx].real * y[idx].real - x[idx].imag * y[idx].imag;
    z[idx].imag = x[idx].real * y[idx].imag - x[idx].imag * y[idx].real;
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
  }else{
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
  int blockSize = 256;
  int gridSize = (length + blockSize - 1) / blockSize;
  switch(dtype){
    case BOOL: add_bool_kernel<<<gridSize, blockSize>>>((bool *)tx->buf, (bool *)ty->buf, (bool *)tz->buf, length); break;
    case INT8: add_tensor_kernel<int8><<<gridSize, blockSize>>>((int8 *)tx->buf, (int8 *)ty->buf, (int8 *)tz->buf, length); break;
    case UINT8: add_tensor_kernel<uint8><<<gridSize, blockSize>>>((uint8 *)tx->buf, (uint8 *)ty->buf, (uint8 *)tz->buf, length); break;
    case INT16: add_tensor_kernel<int16><<<gridSize, blockSize>>>((int16 *)tx->buf, (int16 *)ty->buf, (int16 *)tz->buf, length); break;
    case UINT16: add_tensor_kernel<uint16><<<gridSize, blockSize>>>((uint16 *)tx->buf, (uint16 *)ty->buf, (uint16 *)tz->buf, length); break;
    case INT32: add_tensor_kernel<int32><<<gridSize, blockSize>>>((int32 *)tx->buf, (int32 *)ty->buf, (int32 *)tz->buf, length); break;
    case UINT32: add_tensor_kernel<uint32><<<gridSize, blockSize>>>((uint32 *)tx->buf, (uint32 *)ty->buf, (uint32 *)tz->buf, length); break;
    case INT64: add_tensor_kernel<int64><<<gridSize, blockSize>>>((int64 *)tx->buf, (int64 *)ty->buf, (int64 *)tz->buf, length); break;
    case UINT64: add_tensor_kernel<uint64><<<gridSize, blockSize>>>((uint64 *)tx->buf, (uint64 *)ty->buf, (uint64 *)tz->buf, length); break;
    case FP32: add_tensor_kernel<float32><<<gridSize, blockSize>>>((float32 *)tx->buf, (float32 *)ty->buf, (float32 *)tz->buf, length); break;
    case FP64: add_tensor_kernel<float64><<<gridSize, blockSize>>>((float64 *)tx->buf, (float64 *)ty->buf, (float64 *)tz->buf, length); break;
    case FP128: add_tensor_kernel<float128><<<gridSize, blockSize>>>((float128 *)tx->buf, (float128 *)ty->buf, (float128 *)tz->buf, length); break;
    case CMPX64: add_complex64_kernel<<<gridSize, blockSize>>>((complex64 *)tx->buf, (complex64 *)ty->buf, (complex64 *)tz->buf, length); break;
    case CMPX128: add_complex128_kernel<<<gridSize, blockSize>>>((complex128 *)tx->buf, (complex128 *)ty->buf, (complex128 *)tz->buf, length); break;
    case ERROR:
      PyErr_SetString(PyExc_RuntimeError, "something is wrong i can feel it");
      return NULL;
  }
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA kernel launch failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  err = cudaDeviceSynchronize();
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA synchronization failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  return PyCapsule_New(tz, "tensor_t on CUDA", capsule_destroyer);
}

static PyObject *__sub_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  int blockSize = 256;
  int gridSize = (length + blockSize - 1) / blockSize;
  switch(dtype){
    case BOOL: sub_bool_kernel<<<gridSize, blockSize>>>((bool *)tx->buf, (bool *)ty->buf, (bool *)tz->buf, length); break;
    case INT8: sub_tensor_kernel<int8><<<gridSize, blockSize>>>((int8 *)tx->buf, (int8 *)ty->buf, (int8 *)tz->buf, length); break;
    case UINT8: sub_tensor_kernel<uint8><<<gridSize, blockSize>>>((uint8 *)tx->buf, (uint8 *)ty->buf, (uint8 *)tz->buf, length); break;
    case INT16: sub_tensor_kernel<int16><<<gridSize, blockSize>>>((int16 *)tx->buf, (int16 *)ty->buf, (int16 *)tz->buf, length); break;
    case UINT16: sub_tensor_kernel<uint16><<<gridSize, blockSize>>>((uint16 *)tx->buf, (uint16 *)ty->buf, (uint16 *)tz->buf, length); break;
    case INT32: sub_tensor_kernel<int32><<<gridSize, blockSize>>>((int32 *)tx->buf, (int32 *)ty->buf, (int32 *)tz->buf, length); break;
    case UINT32:sub_tensor_kernel<uint32><<<gridSize, blockSize>>>((uint32 *)tx->buf, (uint32 *)ty->buf, (uint32 *)tz->buf, length); break;
    case INT64: sub_tensor_kernel<int64><<<gridSize, blockSize>>>((int64 *)tx->buf, (int64 *)ty->buf, (int64 *)tz->buf, length); break;
    case UINT64: sub_tensor_kernel<uint64><<<gridSize, blockSize>>>((uint64 *)tx->buf, (uint64 *)ty->buf, (uint64 *)tz->buf, length); break;
    case FP32: sub_tensor_kernel<float32><<<gridSize, blockSize>>>((float32 *)tx->buf, (float32 *)ty->buf, (float32 *)tz->buf, length); break;
    case FP64: sub_tensor_kernel<float64><<<gridSize, blockSize>>>((float64 *)tx->buf, (float64 *)ty->buf, (float64 *)tz->buf, length); break;
    case FP128: sub_tensor_kernel<float128><<<gridSize, blockSize>>>((float128 *)tx->buf, (float128 *)ty->buf, (float128 *)tz->buf, length); break;
    case CMPX64: sub_complex64_kernel<<<gridSize, blockSize>>>((complex64 *)tx->buf, (complex64 *)ty->buf, (complex64 *)tz->buf, length); break;
    case CMPX128: sub_complex128_kernel<<<gridSize, blockSize>>>((complex128 *)tx->buf, (complex128 *)ty->buf, (complex128 *)tz->buf, length); break;
    case ERROR:
      PyErr_SetString(PyExc_RuntimeError, "something is wrong i can feel it");
      return NULL;
  }
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA kernel launch failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  err = cudaDeviceSynchronize();
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA synchronization failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  return PyCapsule_New(tz, "tensor_t on CUDA", capsule_destroyer);
}

static PyObject *__mul_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  int blockSize = 256;
  int gridSize = (length + blockSize - 1) / blockSize;
  switch(dtype){
    case BOOL: mul_bool_kernel<<<gridSize, blockSize>>>((bool *)tx->buf, (bool *)ty->buf, (bool *)tz->buf, length); break;
    case INT8: mul_tensor_kernel<int8><<<gridSize, blockSize>>>((int8 *)tx->buf, (int8 *)ty->buf, (int8 *)tz->buf, length); break;
    case UINT8: mul_tensor_kernel<uint8><<<gridSize, blockSize>>>((uint8 *)tx->buf, (uint8 *)ty->buf, (uint8 *)tz->buf, length); break;
    case INT16: mul_tensor_kernel<int16><<<gridSize, blockSize>>>((int16 *)tx->buf, (int16 *)ty->buf, (int16 *)tz->buf, length); break;
    case UINT16: mul_tensor_kernel<uint16><<<gridSize, blockSize>>>((uint16 *)tx->buf, (uint16 *)ty->buf, (uint16 *)tz->buf, length); break;
    case INT32: mul_tensor_kernel<int32><<<gridSize, blockSize>>>((int32 *)tx->buf, (int32 *)ty->buf, (int32 *)tz->buf, length); break;
    case UINT32: mul_tensor_kernel<uint32><<<gridSize, blockSize>>>((uint32 *)tx->buf, (uint32 *)ty->buf, (uint32 *)tz->buf, length); break;
    case INT64: mul_tensor_kernel<int64><<<gridSize, blockSize>>>((int64 *)tx->buf, (int64 *)ty->buf, (int64 *)tz->buf, length); break;
    case UINT64: mul_tensor_kernel<uint64><<<gridSize, blockSize>>>((uint64 *)tx->buf, (uint64 *)ty->buf, (uint64 *)tz->buf, length); break;
    case FP32: mul_tensor_kernel<float32><<<gridSize, blockSize>>>((float32 *)tx->buf, (float32 *)ty->buf, (float32 *)tz->buf, length); break;
    case FP64: mul_tensor_kernel<float64><<<gridSize, blockSize>>>((float64 *)tx->buf, (float64 *)ty->buf, (float64 *)tz->buf, length); break;
    case FP128: mul_tensor_kernel<float128><<<gridSize, blockSize>>>((float128 *)tx->buf, (float128 *)ty->buf, (float128 *)tz->buf, length); break;
    case CMPX64: mul_complex64_kernel<<<gridSize, blockSize>>>((complex64 *)tx->buf, (complex64 *)ty->buf, (complex64 *)tz->buf, length); break;
    case CMPX128: mul_complex128_kernel<<<gridSize, blockSize>>>((complex128 *)tx->buf, (complex128 *)ty->buf, (complex128 *)tz->buf, length); break;
    case ERROR:
      PyErr_SetString(PyExc_RuntimeError, "something is wrong i can feel it");
      return NULL;
  }
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA kernel launch failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  err = cudaDeviceSynchronize();
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA synchronization failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  return PyCapsule_New(tz, "tensor_t on CUDA", capsule_destroyer);
}

static PyObject *__eq_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  int blockSize = 256;
  int gridSize = (length + blockSize - 1) / blockSize;
  switch(dtype){
    case BOOL: eq_tensor_kernel<<<gridSize, blockSize>>>((bool *)tx->buf, (bool *)ty->buf, (bool *)tz->buf, length); break;
    case INT8: eq_tensor_kernel<int8><<<gridSize, blockSize>>>((int8 *)tx->buf, (int8 *)ty->buf, (bool *)tz->buf, length); break;
    case UINT8: eq_tensor_kernel<uint8><<<gridSize, blockSize>>>((uint8 *)tx->buf, (uint8 *)ty->buf, (bool *)tz->buf, length); break;
    case INT16: eq_tensor_kernel<int16><<<gridSize, blockSize>>>((int16 *)tx->buf, (int16 *)ty->buf, (bool *)tz->buf, length); break;
    case UINT16: eq_tensor_kernel<uint16><<<gridSize, blockSize>>>((uint16 *)tx->buf, (uint16 *)ty->buf, (bool *)tz->buf, length); break;
    case INT32: eq_tensor_kernel<int32><<<gridSize, blockSize>>>((int32 *)tx->buf, (int32 *)ty->buf, (bool *)tz->buf, length); break;
    case UINT32: eq_tensor_kernel<uint32><<<gridSize, blockSize>>>((uint32 *)tx->buf, (uint32 *)ty->buf, (bool *)tz->buf, length); break;
    case INT64: eq_tensor_kernel<int64><<<gridSize, blockSize>>>((int64 *)tx->buf, (int64 *)ty->buf, (bool *)tz->buf, length); break;
    case UINT64: eq_tensor_kernel<uint64><<<gridSize, blockSize>>>((uint64 *)tx->buf, (uint64 *)ty->buf, (bool *)tz->buf, length); break;
    case FP32: eq_tensor_kernel<float32><<<gridSize, blockSize>>>((float32 *)tx->buf, (float32 *)ty->buf, (bool *)tz->buf, length); break;
    case FP64: eq_tensor_kernel<float64><<<gridSize, blockSize>>>((float64 *)tx->buf, (float64 *)ty->buf, (bool *)tz->buf, length); break;
    case FP128: eq_tensor_kernel<float128><<<gridSize, blockSize>>>((float128 *)tx->buf, (float128 *)ty->buf, (bool *)tz->buf, length); break;
    case CMPX64: eq_cmpx_tensor_kernel<complex64><<<gridSize, blockSize>>>((complex64 *)tx->buf, (complex64 *)ty->buf, (bool *)tz->buf, length); break;
    case CMPX128: eq_cmpx_tensor_kernel<complex128><<<gridSize, blockSize>>>((complex128 *)tx->buf, (complex128 *)ty->buf, (bool *)tz->buf, length); break;
    case ERROR:
      PyErr_SetString(PyExc_RuntimeError, "something is wrong i can feel it");
      return NULL;
  }
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA kernel launch failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  err = cudaDeviceSynchronize();
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA synchronization failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  return PyCapsule_New(tz, "tensor_t on CUDA", capsule_destroyer);
}

static PyObject *__ne_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  int blockSize = 256;
  int gridSize = (length + blockSize - 1) / blockSize;
  switch(dtype){
    case BOOL: ne_tensor_kernel<<<gridSize, blockSize>>>((bool *)tx->buf, (bool *)ty->buf, (bool *)tz->buf, length); break;
    case INT8: ne_tensor_kernel<int8><<<gridSize, blockSize>>>((int8 *)tx->buf, (int8 *)ty->buf, (bool *)tz->buf, length); break;
    case UINT8: ne_tensor_kernel<uint8><<<gridSize, blockSize>>>((uint8 *)tx->buf, (uint8 *)ty->buf, (bool *)tz->buf, length); break;
    case INT16: ne_tensor_kernel<int16><<<gridSize, blockSize>>>((int16 *)tx->buf, (int16 *)ty->buf, (bool *)tz->buf, length); break;
    case UINT16: ne_tensor_kernel<uint16><<<gridSize, blockSize>>>((uint16 *)tx->buf, (uint16 *)ty->buf, (bool *)tz->buf, length); break;
    case INT32: ne_tensor_kernel<int32><<<gridSize, blockSize>>>((int32 *)tx->buf, (int32 *)ty->buf, (bool *)tz->buf, length); break;
    case UINT32: ne_tensor_kernel<uint32><<<gridSize, blockSize>>>((uint32 *)tx->buf, (uint32 *)ty->buf, (bool *)tz->buf, length); break;
    case INT64: ne_tensor_kernel<int64><<<gridSize, blockSize>>>((int64 *)tx->buf, (int64 *)ty->buf, (bool *)tz->buf, length); break;
    case UINT64: ne_tensor_kernel<uint64><<<gridSize, blockSize>>>((uint64 *)tx->buf, (uint64 *)ty->buf, (bool *)tz->buf, length); break;
    case FP32: ne_tensor_kernel<float32><<<gridSize, blockSize>>>((float32 *)tx->buf, (float32 *)ty->buf, (bool *)tz->buf, length); break;
    case FP64: ne_tensor_kernel<float64><<<gridSize, blockSize>>>((float64 *)tx->buf, (float64 *)ty->buf, (bool *)tz->buf, length); break;
    case FP128: ne_tensor_kernel<float128><<<gridSize, blockSize>>>((float128 *)tx->buf, (float128 *)ty->buf, (bool *)tz->buf, length); break;
    case CMPX64: ne_cmpx_tensor_kernel<complex64><<<gridSize, blockSize>>>((complex64 *)tx->buf, (complex64 *)ty->buf, (bool *)tz->buf, length); break;
    case CMPX128: ne_cmpx_tensor_kernel<complex128><<<gridSize, blockSize>>>((complex128 *)tx->buf, (complex128 *)ty->buf, (bool *)tz->buf, length); break;
    case ERROR:
      PyErr_SetString(PyExc_RuntimeError, "something is wrong i can feel it");
      return NULL;
  }
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA kernel launch failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  err = cudaDeviceSynchronize();
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA synchronization failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  return PyCapsule_New(tz, "tensor_t on CUDA", capsule_destroyer);
}

static PyObject *__gt_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  int blockSize = 256;
  int gridSize = (length + blockSize - 1) / blockSize;
  switch(dtype){
    case BOOL: gt_tensor_kernel<<<gridSize, blockSize>>>((bool *)tx->buf, (bool *)ty->buf, (bool *)tz->buf, length); break;
    case INT8: gt_tensor_kernel<int8><<<gridSize, blockSize>>>((int8 *)tx->buf, (int8 *)ty->buf, (bool *)tz->buf, length); break;
    case UINT8: gt_tensor_kernel<uint8><<<gridSize, blockSize>>>((uint8 *)tx->buf, (uint8 *)ty->buf, (bool *)tz->buf, length); break;
    case INT16: gt_tensor_kernel<int16><<<gridSize, blockSize>>>((int16 *)tx->buf, (int16 *)ty->buf, (bool *)tz->buf, length); break;
    case UINT16: gt_tensor_kernel<uint16><<<gridSize, blockSize>>>((uint16 *)tx->buf, (uint16 *)ty->buf, (bool *)tz->buf, length); break;
    case INT32: gt_tensor_kernel<int32><<<gridSize, blockSize>>>((int32 *)tx->buf, (int32 *)ty->buf, (bool *)tz->buf, length); break;
    case UINT32: gt_tensor_kernel<uint32><<<gridSize, blockSize>>>((uint32 *)tx->buf, (uint32 *)ty->buf, (bool *)tz->buf, length); break;
    case INT64: gt_tensor_kernel<int64><<<gridSize, blockSize>>>((int64 *)tx->buf, (int64 *)ty->buf, (bool *)tz->buf, length); break;
    case UINT64: gt_tensor_kernel<uint64><<<gridSize, blockSize>>>((uint64 *)tx->buf, (uint64 *)ty->buf, (bool *)tz->buf, length); break;
    case FP32: gt_tensor_kernel<float32><<<gridSize, blockSize>>>((float32 *)tx->buf, (float32 *)ty->buf, (bool *)tz->buf, length); break;
    case FP64: gt_tensor_kernel<float64><<<gridSize, blockSize>>>((float64 *)tx->buf, (float64 *)ty->buf, (bool *)tz->buf, length); break;
    case FP128: gt_tensor_kernel<float128><<<gridSize, blockSize>>>((float128 *)tx->buf, (float128 *)ty->buf, (bool *)tz->buf, length); break;
    case ERROR:
      PyErr_SetString(PyExc_RuntimeError, "something is wrong i can feel it");
      return NULL;
  }
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA kernel launch failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  err = cudaDeviceSynchronize();
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA synchronization failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  return PyCapsule_New(tz, "tensor_t on CUDA", capsule_destroyer);
}

static PyObject *__ge_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  int blockSize = 256;
  int gridSize = (length + blockSize - 1) / blockSize;
  switch(dtype){
    case BOOL: ge_tensor_kernel<<<gridSize, blockSize>>>((bool *)tx->buf, (bool *)ty->buf, (bool *)tz->buf, length); break;
    case INT8: ge_tensor_kernel<int8><<<gridSize, blockSize>>>((int8 *)tx->buf, (int8 *)ty->buf, (bool *)tz->buf, length); break;
    case UINT8: ge_tensor_kernel<uint8><<<gridSize, blockSize>>>((uint8 *)tx->buf, (uint8 *)ty->buf, (bool *)tz->buf, length); break;
    case INT16: ge_tensor_kernel<int16><<<gridSize, blockSize>>>((int16 *)tx->buf, (int16 *)ty->buf, (bool *)tz->buf, length); break;
    case UINT16: ge_tensor_kernel<uint16><<<gridSize, blockSize>>>((uint16 *)tx->buf, (uint16 *)ty->buf, (bool *)tz->buf, length); break;
    case INT32: ge_tensor_kernel<int32><<<gridSize, blockSize>>>((int32 *)tx->buf, (int32 *)ty->buf, (bool *)tz->buf, length); break;
    case UINT32: ge_tensor_kernel<uint32><<<gridSize, blockSize>>>((uint32 *)tx->buf, (uint32 *)ty->buf, (bool *)tz->buf, length); break;
    case INT64: ge_tensor_kernel<int64><<<gridSize, blockSize>>>((int64 *)tx->buf, (int64 *)ty->buf, (bool *)tz->buf, length); break;
    case UINT64: ge_tensor_kernel<uint64><<<gridSize, blockSize>>>((uint64 *)tx->buf, (uint64 *)ty->buf, (bool *)tz->buf, length); break;
    case FP32: ge_tensor_kernel<float32><<<gridSize, blockSize>>>((float32 *)tx->buf, (float32 *)ty->buf, (bool *)tz->buf, length); break;
    case FP64: ge_tensor_kernel<float64><<<gridSize, blockSize>>>((float64 *)tx->buf, (float64 *)ty->buf, (bool *)tz->buf, length); break;
    case FP128: ge_tensor_kernel<float128><<<gridSize, blockSize>>>((float128 *)tx->buf, (float128 *)ty->buf, (bool *)tz->buf, length); break;
    case ERROR:
      PyErr_SetString(PyExc_RuntimeError, "something is wrong i can feel it");
      return NULL;
  }
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA kernel launch failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  err = cudaDeviceSynchronize();
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA synchronization failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  return PyCapsule_New(tz, "tensor_t on CUDA", capsule_destroyer);
}

static PyObject *__lt_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  int blockSize = 256;
  int gridSize = (length + blockSize - 1) / blockSize;
  switch(dtype){
    case BOOL: lt_tensor_kernel<<<gridSize, blockSize>>>((bool *)tx->buf, (bool *)ty->buf, (bool *)tz->buf, length); break;
    case INT8: lt_tensor_kernel<int8><<<gridSize, blockSize>>>((int8 *)tx->buf, (int8 *)ty->buf, (bool *)tz->buf, length); break;
    case UINT8: lt_tensor_kernel<uint8><<<gridSize, blockSize>>>((uint8 *)tx->buf, (uint8 *)ty->buf, (bool *)tz->buf, length); break;
    case INT16: lt_tensor_kernel<int16><<<gridSize, blockSize>>>((int16 *)tx->buf, (int16 *)ty->buf, (bool *)tz->buf, length); break;
    case UINT16: lt_tensor_kernel<uint16><<<gridSize, blockSize>>>((uint16 *)tx->buf, (uint16 *)ty->buf, (bool *)tz->buf, length); break;
    case INT32: lt_tensor_kernel<int32><<<gridSize, blockSize>>>((int32 *)tx->buf, (int32 *)ty->buf, (bool *)tz->buf, length); break;
    case UINT32: lt_tensor_kernel<uint32><<<gridSize, blockSize>>>((uint32 *)tx->buf, (uint32 *)ty->buf, (bool *)tz->buf, length); break;
    case INT64: lt_tensor_kernel<int64><<<gridSize, blockSize>>>((int64 *)tx->buf, (int64 *)ty->buf, (bool *)tz->buf, length); break;
    case UINT64: lt_tensor_kernel<uint64><<<gridSize, blockSize>>>((uint64 *)tx->buf, (uint64 *)ty->buf, (bool *)tz->buf, length); break;
    case FP32: lt_tensor_kernel<float32><<<gridSize, blockSize>>>((float32 *)tx->buf, (float32 *)ty->buf, (bool *)tz->buf, length); break;
    case FP64: lt_tensor_kernel<float64><<<gridSize, blockSize>>>((float64 *)tx->buf, (float64 *)ty->buf, (bool *)tz->buf, length); break;
    case FP128: lt_tensor_kernel<float128><<<gridSize, blockSize>>>((float128 *)tx->buf, (float128 *)ty->buf, (bool *)tz->buf, length); break;
    case ERROR:
      PyErr_SetString(PyExc_RuntimeError, "something is wrong i can feel it");
      return NULL;
  }
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA kernel launch failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  err = cudaDeviceSynchronize();
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA synchronization failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  return PyCapsule_New(tz, "tensor_t on CUDA", capsule_destroyer);
}

static PyObject *__le_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  int blockSize = 256;
  int gridSize = (length + blockSize - 1) / blockSize;
  switch(dtype){
    case BOOL: le_tensor_kernel<<<gridSize, blockSize>>>((bool *)tx->buf, (bool *)ty->buf, (bool *)tz->buf, length); break;
    case INT8: le_tensor_kernel<int8><<<gridSize, blockSize>>>((int8 *)tx->buf, (int8 *)ty->buf, (bool *)tz->buf, length); break;
    case UINT8: le_tensor_kernel<uint8><<<gridSize, blockSize>>>((uint8 *)tx->buf, (uint8 *)ty->buf, (bool *)tz->buf, length); break;
    case INT16: le_tensor_kernel<int16><<<gridSize, blockSize>>>((int16 *)tx->buf, (int16 *)ty->buf, (bool *)tz->buf, length); break;
    case UINT16: le_tensor_kernel<uint16><<<gridSize, blockSize>>>((uint16 *)tx->buf, (uint16 *)ty->buf, (bool *)tz->buf, length); break;
    case INT32: le_tensor_kernel<int32><<<gridSize, blockSize>>>((int32 *)tx->buf, (int32 *)ty->buf, (bool *)tz->buf, length); break;
    case UINT32: le_tensor_kernel<uint32><<<gridSize, blockSize>>>((uint32 *)tx->buf, (uint32 *)ty->buf, (bool *)tz->buf, length); break;
    case INT64: le_tensor_kernel<int64><<<gridSize, blockSize>>>((int64 *)tx->buf, (int64 *)ty->buf, (bool *)tz->buf, length); break;
    case UINT64: le_tensor_kernel<uint64><<<gridSize, blockSize>>>((uint64 *)tx->buf, (uint64 *)ty->buf, (bool *)tz->buf, length); break;
    case FP32: le_tensor_kernel<float32><<<gridSize, blockSize>>>((float32 *)tx->buf, (float32 *)ty->buf, (bool *)tz->buf, length); break;
    case FP64: le_tensor_kernel<float64><<<gridSize, blockSize>>>((float64 *)tx->buf, (float64 *)ty->buf, (bool *)tz->buf, length); break;
    case FP128: le_tensor_kernel<float128><<<gridSize, blockSize>>>((float128 *)tx->buf, (float128 *)ty->buf, (bool *)tz->buf, length); break;
    case ERROR:
      PyErr_SetString(PyExc_RuntimeError, "something is wrong i can feel it");
      return NULL;
  }
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA kernel launch failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  err = cudaDeviceSynchronize();
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA synchronization failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  return PyCapsule_New(tz, "tensor_t on CUDA", capsule_destroyer);
}

static PyObject *__neg_tensor__(const tensor_t *tx, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  int blockSize = 256;
  int gridSize = (length * blockSize - 1) / blockSize;
  switch(dtype){
    case INT8: neg_tensor_kernel<int8><<<gridSize, blockSize>>>((int8 *)tx->buf, (int8 *)tz->buf, length); break;
    case UINT8: neg_tensor_kernel<uint8><<<gridSize, blockSize>>>((uint8 *)tx->buf, (uint8 *)tz->buf, length); break;
    case INT16: neg_tensor_kernel<int16><<<gridSize, blockSize>>>((int16 *)tx->buf, (int16 *)tz->buf, length); break;
    case UINT16: neg_tensor_kernel<uint16><<<gridSize, blockSize>>>((uint16 *)tx->buf, (uint16 *)tz->buf, length); break;
    case INT32: neg_tensor_kernel<int32><<<gridSize, blockSize>>>((int32 *)tx->buf, (int32 *)tz->buf, length); break;
    case UINT32: neg_tensor_kernel<uint32><<<gridSize, blockSize>>>((uint32 *)tx->buf, (uint32 *)tz->buf, length); break;
    case INT64: neg_tensor_kernel<int64><<<gridSize, blockSize>>>((int64 *)tx->buf, (int64 *)tz->buf, length); break;
    case UINT64: neg_tensor_kernel<uint64><<<gridSize, blockSize>>>((uint64 *)tx->buf, (uint64 *)tz->buf, length); break;
    case FP32: neg_tensor_kernel<float32><<<gridSize, blockSize>>>((float32 *)tx->buf, (float32 *)tz->buf, length); break;
    case FP64: neg_tensor_kernel<float64><<<gridSize, blockSize>>>((float64 *)tx->buf, (float64 *)tz->buf, length); break;
    case CMPX64: neg_cmpx_tensor_kernel<complex64><<<gridSize, blockSize>>>((complex64 *)tx->buf, (complex64 *)tz->buf, length); break;
    case CMPX128: neg_cmpx_tensor_kernel<complex128><<<gridSize, blockSize>>>((complex128 *)tx->buf, (complex128 *)tz->buf, length); break;
  }
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA kernel launch failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  err = cudaDeviceSynchronize();
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA synchronization failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  return PyCapsule_New(tz, "tensor_t on CUDA", capsule_destroyer);
}

static PyObject *__pos_tensor__(const tensor_t *tx, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  int blockSize = 256;
  int gridSize = (length * blockSize - 1) / blockSize;
  switch(dtype){
    case INT8: pos_tensor_kernel<int8><<<gridSize, blockSize>>>((int8 *)tx->buf, (int8 *)tz->buf, length); break;
    case UINT8: pos_tensor_kernel<uint8><<<gridSize, blockSize>>>((uint8 *)tx->buf, (uint8 *)tz->buf, length); break;
    case INT16: pos_tensor_kernel<int16><<<gridSize, blockSize>>>((int16 *)tx->buf, (int16 *)tz->buf, length); break;
    case UINT16: pos_tensor_kernel<uint16><<<gridSize, blockSize>>>((uint16 *)tx->buf, (uint16 *)tz->buf, length); break;
    case INT32: pos_tensor_kernel<int32><<<gridSize, blockSize>>>((int32 *)tx->buf, (int32 *)tz->buf, length); break;
    case UINT32: pos_tensor_kernel<uint32><<<gridSize, blockSize>>>((uint32 *)tx->buf, (uint32 *)tz->buf, length); break;
    case INT64: pos_tensor_kernel<int64><<<gridSize, blockSize>>>((int64 *)tx->buf, (int64 *)tz->buf, length); break;
    case UINT64: pos_tensor_kernel<uint64><<<gridSize, blockSize>>>((uint64 *)tx->buf, (uint64 *)tz->buf, length); break;
    case FP32: pos_tensor_kernel<float32><<<gridSize, blockSize>>>((float32 *)tx->buf, (float32 *)tz->buf, length); break;
    case FP64: pos_tensor_kernel<float64><<<gridSize, blockSize>>>((float64 *)tx->buf, (float64 *)tz->buf, length); break;
    case CMPX64: pos_cmpx_tensor_kernel<complex64><<<gridSize, blockSize>>>((complex64 *)tx->buf, (complex64 *)tz->buf, length); break;
    case CMPX128: pos_cmpx_tensor_kernel<complex128><<<gridSize, blockSize>>>((complex128 *)tx->buf, (complex128 *)tz->buf, length); break;
  }
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA kernel launch failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  err = cudaDeviceSynchronize();
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA synchronization failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  return PyCapsule_New(tz, "tensor_t on CUDA", capsule_destroyer);
}

static PyObject *__abs_tensor__(const tensor_t *tx, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  int blockSize = 256;
  int gridSize = (length * blockSize - 1) / blockSize;
  switch(dtype){
    case INT8: abs_tensor_kernel<int8><<<gridSize, blockSize>>>((int8 *)tx->buf, (int8 *)tz->buf, length); break;
    case UINT8: abs_tensor_kernel<uint8><<<gridSize, blockSize>>>((uint8 *)tx->buf, (uint8 *)tz->buf, length); break;
    case INT16: abs_tensor_kernel<int16><<<gridSize, blockSize>>>((int16 *)tx->buf, (int16 *)tz->buf, length); break;
    case UINT16: abs_tensor_kernel<uint16><<<gridSize, blockSize>>>((uint16 *)tx->buf, (uint16 *)tz->buf, length); break;
    case INT32: abs_tensor_kernel<int32><<<gridSize, blockSize>>>((int32 *)tx->buf, (int32 *)tz->buf, length); break;
    case UINT32: abs_tensor_kernel<uint32><<<gridSize, blockSize>>>((uint32 *)tx->buf, (uint32 *)tz->buf, length); break;
    case INT64: abs_tensor_kernel<int64><<<gridSize, blockSize>>>((int64 *)tx->buf, (int64 *)tz->buf, length); break;
    case UINT64: abs_tensor_kernel<uint64><<<gridSize, blockSize>>>((uint64 *)tx->buf, (uint64 *)tz->buf, length); break;
    case FP32: abs_tensor_kernel<float32><<<gridSize, blockSize>>>((float32 *)tx->buf, (float32 *)tz->buf, length); break;
    case FP64: abs_tensor_kernel<float64><<<gridSize, blockSize>>>((float64 *)tx->buf, (float64 *)tz->buf, length); break;
    case CMPX64: abs_cmpx_tensor_kernel<complex64, float32><<<gridSize, blockSize>>>((complex64 *)tx->buf, (float32 *)tz->buf, length); break;
    case CMPX128: abs_cmpx_tensor_kernel<complex128, float64><<<gridSize, blockSize>>>((complex128 *)tx->buf, (float64 *)tz->buf, length); break;
  }
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA kernel launch failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  err = cudaDeviceSynchronize();
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA synchronization failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  return PyCapsule_New(tz, "tensor_t on CUDA", capsule_destroyer);
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
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __add_tensor__(tx, ty, tz);
}
static PyObject *sub(PyObject *self, PyObject *args){
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
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __sub_tensor__(tx, ty, tz);
}

static PyObject *mul(PyObject *self, PyObject *args){
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
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __mul_tensor__(tx, ty, tz);
}

static PyObject *eq(PyObject *self, PyObject *args){
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
  tensor_t *tz = (tensor_t *)malloc(sizeof(tensor_t));
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
    return NULL;
  }
  tz->dtype = BOOL;
  tz->size = tx->size;
  tz->ndim = tx->ndim;
  tz->element_size = sizeof(bool);
  tz->device = tx->device;
  tz->stride = NULL;
  if(tx->ndim > 0 && tx->shape){
    tz->shape = (size_t *)malloc(sizeof(size_t) * tz->ndim);
    if(!tz->shape){
      free(tz);
      PyErr_SetString(PyExc_RuntimeError, "tensor_t.shape allocation failed!");
      return NULL;
    }
    for(size_t i = 0; i < tz->ndim; i++){
      tz->shape[i] = tx->shape[i];
    }
  }else{
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
  cudaError_t err = cudaMalloc(&tz->storage->ptr, tz->storage->bytes);
  if(err != cudaSuccess){
    if(tz->shape) free(tz->shape);
    free(tz->storage);
    free(tz);
    PyErr_Format(PyExc_RuntimeError, "CUDA malloc failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  tz->buf = tz->storage->ptr;
  return __eq_tensor__(tx, ty, tz);
}

static PyObject *ne(PyObject *self, PyObject *args){
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
  tensor_t *tz = (tensor_t *)malloc(sizeof(tensor_t));
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
    return NULL;
  }
  tz->dtype = BOOL;
  tz->size = tx->size;
  tz->ndim = tx->ndim;
  tz->element_size = sizeof(bool);
  tz->device = tx->device;
  tz->stride = NULL;
  if(tx->ndim > 0 && tx->shape){
    tz->shape = (size_t *)malloc(sizeof(size_t) * tz->ndim);
    if(!tz->shape){
      free(tz);
      PyErr_SetString(PyExc_RuntimeError, "tensor_t.shape allocation failed!");
      return NULL;
    }
    for(size_t i = 0; i < tz->ndim; i++){
      tz->shape[i] = tx->shape[i];
    }
  }else{
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
  cudaError_t err = cudaMalloc(&tz->storage->ptr, tz->storage->bytes);
  if(err != cudaSuccess){
    if(tz->shape) free(tz->shape);
    free(tz->storage);
    free(tz);
    PyErr_Format(PyExc_RuntimeError, "CUDA malloc failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  tz->buf = tz->storage->ptr;
  return __ne_tensor__(tx, ty, tz);
}

static PyObject *gt(PyObject *self, PyObject *args){
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
  tensor_t *tz = (tensor_t *)malloc(sizeof(tensor_t));
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
    return NULL;
  }
  tz->dtype = BOOL;
  tz->size = tx->size;
  tz->ndim = tx->ndim;
  tz->element_size = sizeof(bool);
  tz->device = tx->device;
  tz->stride = NULL;
  if(tx->ndim > 0 && tx->shape){
    tz->shape = (size_t *)malloc(sizeof(size_t) * tz->ndim);
    if(!tz->shape){
      free(tz);
      PyErr_SetString(PyExc_RuntimeError, "tensor_t.shape allocation failed!");
      return NULL;
    }
    for(size_t i = 0; i < tz->ndim; i++){
      tz->shape[i] = tx->shape[i];
    }
  }else{
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
  cudaError_t err = cudaMalloc(&tz->storage->ptr, tz->storage->bytes);
  if(err != cudaSuccess){
    if(tz->shape) free(tz->shape);
    free(tz->storage);
    free(tz);
    PyErr_Format(PyExc_RuntimeError, "CUDA malloc failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  tz->buf = tz->storage->ptr;
  return __gt_tensor__(tx, ty, tz);
}

static PyObject *ge(PyObject *self, PyObject *args){
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
  tensor_t *tz = (tensor_t *)malloc(sizeof(tensor_t));
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
    return NULL;
  }
  tz->dtype = BOOL;
  tz->size = tx->size;
  tz->ndim = tx->ndim;
  tz->element_size = sizeof(bool);
  tz->device = tx->device;
  tz->stride = NULL;
  if(tx->ndim > 0 && tx->shape){
    tz->shape = (size_t *)malloc(sizeof(size_t) * tz->ndim);
    if(!tz->shape){
      free(tz);
      PyErr_SetString(PyExc_RuntimeError, "tensor_t.shape allocation failed!");
      return NULL;
    }
    for(size_t i = 0; i < tz->ndim; i++){
      tz->shape[i] = tx->shape[i];
    }
  }else{
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
  cudaError_t err = cudaMalloc(&tz->storage->ptr, tz->storage->bytes);
  if(err != cudaSuccess){
    if(tz->shape) free(tz->shape);
    free(tz->storage);
    free(tz);
    PyErr_Format(PyExc_RuntimeError, "CUDA malloc failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  tz->buf = tz->storage->ptr;
  return __ge_tensor__(tx, ty, tz);
}

static PyObject *lt(PyObject *self, PyObject *args){
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
  tensor_t *tz = (tensor_t *)malloc(sizeof(tensor_t));
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
    return NULL;
  }
  tz->dtype = BOOL;
  tz->size = tx->size;
  tz->ndim = tx->ndim;
  tz->element_size = sizeof(bool);
  tz->device = tx->device;
  tz->stride = NULL;
  if(tx->ndim > 0 && tx->shape){
    tz->shape = (size_t *)malloc(sizeof(size_t) * tz->ndim);
    if(!tz->shape){
      free(tz);
      PyErr_SetString(PyExc_RuntimeError, "tensor_t.shape allocation failed!");
      return NULL;
    }
    for(size_t i = 0; i < tz->ndim; i++){
      tz->shape[i] = tx->shape[i];
    }
  }else{
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
  cudaError_t err = cudaMalloc(&tz->storage->ptr, tz->storage->bytes);
  if(err != cudaSuccess){
    if(tz->shape) free(tz->shape);
    free(tz->storage);
    free(tz);
    PyErr_Format(PyExc_RuntimeError, "CUDA malloc failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  tz->buf = tz->storage->ptr;
  return __lt_tensor__(tx, ty, tz);
}

static PyObject *le(PyObject *self, PyObject *args){
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
  tensor_t *tz = (tensor_t *)malloc(sizeof(tensor_t));
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
    return NULL;
  }
  tz->dtype = BOOL;
  tz->size = tx->size;
  tz->ndim = tx->ndim;
  tz->element_size = sizeof(bool);
  tz->device = tx->device;
  tz->stride = NULL;
  if(tx->ndim > 0 && tx->shape){
    tz->shape = (size_t *)malloc(sizeof(size_t) * tz->ndim);
    if(!tz->shape){
      free(tz);
      PyErr_SetString(PyExc_RuntimeError, "tensor_t.shape allocation failed!");
      return NULL;
    }
    for(size_t i = 0; i < tz->ndim; i++){
      tz->shape[i] = tx->shape[i];
    }
  }else{
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
  cudaError_t err = cudaMalloc(&tz->storage->ptr, tz->storage->bytes);
  if(err != cudaSuccess){
    if(tz->shape) free(tz->shape);
    free(tz->storage);
    free(tz);
    PyErr_Format(PyExc_RuntimeError, "CUDA malloc failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  tz->buf = tz->storage->ptr;
  return __le_tensor__(tx, ty, tz);
}

static PyObject *neg(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor_t capsule pointer");
    return NULL;
  }
  tensor_t *tx = (tensor_t *)PyCapsule_GetPointer(x, "tensor_t on CUDA");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  if(tx->device.type != CUDA){
    PyErr_Format(PyExc_RuntimeError, "Invalid device: %s, tensor_t must be on CUDA device", tx->device.type);
    return NULL;
  }
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __neg_tensor__(tx, tz);
}

static PyObject *pos(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor_t capsule pointer");
    return NULL;
  }
  tensor_t *tx = (tensor_t *)PyCapsule_GetPointer(x, "tensor_t on CUDA");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  if(tx->device.type != CUDA){
    PyErr_Format(PyExc_RuntimeError, "Invalid device: %s, tensor_t must be on CUDA device", tx->device.type);
    return NULL;
  }
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __pos_tensor__(tx, tz);
}

static PyObject *abs_(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor_t capsule pointer");
    return NULL;
  }
  tensor_t *tx = (tensor_t *)PyCapsule_GetPointer(x, "tensor_t on CUDA");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  if(tx->device.type != CUDA){
    PyErr_Format(PyExc_RuntimeError, "Invalid device: %s, tensor_t must be on CUDA device", tx->device.type);
    return NULL;
  }
  dtype_t res_dtype = tx->dtype;
  size_t res_elem_size = tx->element_size;
  if(tx->dtype == CMPX64){
    res_dtype = FP32;
    res_elem_size = getsize(FP32);
  }else if(tx->dtype == CMPX128){
    res_dtype = FP64;
    res_elem_size = getsize(FP64);
  }
  tensor_t *tz = alloc_result_tensor(tx);
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
    return NULL;
  }
  tz->dtype = res_dtype;
  tz->element_size = res_elem_size;
  tz->storage->bytes = tz->size * tz->element_size;
  return __abs_tensor__(tx, tz);
}

static PyMethodDef methods[] = {
  {"add", add, METH_VARARGS, "element-wise addition of two tensors or tensor with scalar on CUDA"},
  {"sub", sub, METH_VARARGS, "element-wise subtraction of two tensors or tensor with scalar on CUDA"},
  {"mul", mul, METH_VARARGS, "element-wise multiplication of two tensors or tensor with scalar on CUDA"},
  {"eq", eq, METH_VARARGS, "element-wise equivalence of two tensors or tensor with scalar on CUDA"},
  {"ne", ne, METH_VARARGS, "element-wise non-equivalence of two tensors or tensor with scalar on CUDA"},
  {"gt", gt, METH_VARARGS, "element-wise greater than op of two tensors or tensor with scalar on CUDA"},
  {"ge", ge, METH_VARARGS, "element-wise greater than equal op of two tensors or tensor with scalar on CUDA"},
  {"lt", lt, METH_VARARGS, "element-wise less than op of two tensors or tensor with scalar on CUDA"},
  {"le", le, METH_VARARGS, "element-wise less than equal op of two tensors or tensor with scalar on CUDA"},
  {"neg", neg, METH_VARARGS, "negation operation on tensor on CUDA"},
  {"pos", pos, METH_VARARGS, "non-negation operation on tensor on CUDA"},
  {"abs", abs_, METH_VARARGS, "absolute value of tensor or tensor with scalar on CUDA"},
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
