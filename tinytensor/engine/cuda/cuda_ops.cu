#include <python3.10/Python.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "../tensor.h"

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      PyErr_Format(PyExc_RuntimeError, "CUDA error: %s", cudaGetErrorString(err)); \
      return NULL; \
    } \
  } while(0)

// capsule destroyer
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

// CUDA kernel for element-wise tensor division
template<typename T, typename U>
__global__ void tdiv_tensor_kernel(const T *x, const T *y, U *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = (U)x[idx] / (U)y[idx];
  }
}

// CUDA kernel for element-wise tensor floor division
template<typename T, typename U>
__global__ void fdiv_tensor_kernel(const T *x, const T *y, U *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = (U)(x[idx] / y[idx]);
  }
}

// CUDA kernel for element-wise tensor pow (real numbers)
template<typename T>
__global__ void pow_tensor_kernel(const T *x, const T *y, T *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = (T)pow(x[idx],y[idx]);
  }
}

// CUDA kernel for element-wise complex64 tensor pow
__global__ void pow_cmpx64_tensor_kernel(const complex64 *x, const complex64 *y, complex64 *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    float r = sqrt(x[idx].real * x[idx].real + x[idx].imag * x[idx].imag);
    float theta = atan(x[idx].imag/x[idx].real);
    float A = y[idx].real * log(r) - y[idx].imag * theta;
    float B = y[idx].imag * log(r) + y[idx].real * theta;
    float e = exp(A);
    z[idx].real = e * cos(B);
    z[idx].imag = e * sin(B);
  }
}

// CUDA kernel for element-wise complex128 tensor pow
__global__ void pow_cmpx128_tensor_kernel(const complex128 *x, const complex128 *y, complex128 *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    float r = sqrt(x[idx].real * x[idx].real + x[idx].imag * x[idx].imag);
    float theta = atan(x[idx].imag/x[idx].real);
    float A = y[idx].real * log(r) - y[idx].imag * theta;
    float B = y[idx].imag * log(r) + y[idx].real * theta;
    float e = exp(A);
    z[idx].real = e * cos(B);
    z[idx].imag = e * sin(B);
  }
}

// CUDA kernel for element-wise complex64 tensor floor division
__global__ void tdiv_cmpx64_kernel(const complex64 *x, const complex64 *y, complex64 *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    float a = x[idx].real;
    float b = x[idx].imag;
    float c = y[idx].real;
    float d = y[idx].imag;
    float denom = c*c + d*d;
    if(denom == 0.0f){
      z[idx].real = NAN;
      z[idx].imag = NAN;
      return;
    }
    z[idx].real = (a*c + b*d) / denom;
    z[idx].imag = (b*c - a*d) / denom;
  }
}

// CUDA kernel for element-wise tensor module (real numbers)
template<typename T>
__global__ void mod_tensor_kernel(const T *x, const T *y, T *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = (T)fmodf(x[idx],y[idx]);
  }
}

// CUDA kernel for element-wise complex128 tensor floor division
__global__ void tdiv_cmpx128_kernel(const complex128 *x, const complex128 *y, complex128 *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    float a = x[idx].real;
    float b = x[idx].imag;
    float c = y[idx].real;
    float d = y[idx].imag;
    float denom = c*c + d*d;
    if(denom == 0.0f){
      z[idx].real = NAN;
      z[idx].imag = NAN;
      return;
    }
    z[idx].real = (a*c + b*d) / denom;
    z[idx].imag = (b*c - a*d) / denom;
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

// CUDA kernel for element-wise tensor abs operation
template<typename T>
__global__ void abs_utensor_kernel(const T* x, T*z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = x[idx];
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
__global__ void mul_bool_tensor_kernel(const bool *x, const bool *y, bool *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = x[idx] & y[idx];
  }
}

// CUDA kernel for complex64 addition
__global__ void add_complex64_tensor_kernel(const complex64 *x, const complex64 *y, complex64 *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx].real = x[idx].real + y[idx].real;
    z[idx].imag = x[idx].imag + y[idx].imag;
  }
}

// CUDA kernel for complex64 subtraction
__global__ void sub_complex64_tensor_kernel(const complex64 *x, const complex64 *y, complex64 *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx].real = x[idx].real - y[idx].real;
    z[idx].imag = x[idx].imag - y[idx].imag;
  }
}

// CUDA kernel for complex64 multiplication
__global__ void mul_complex64_tensor_kernel(const complex64 *x, const complex64 *y, complex64 *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx].real = x[idx].real * y[idx].real - x[idx].imag * y[idx].imag;
    z[idx].imag = x[idx].real * y[idx].imag - x[idx].imag * y[idx].real;
  }
}

// CUDA kernel for complex128 addition
__global__ void add_complex128_tensor_kernel(const complex128 *x, const complex128 *y, complex128 *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx].real = x[idx].real + y[idx].real;
    z[idx].imag = x[idx].imag + y[idx].imag;
  }
}

// CUDA kernel for complex128 subtraction
__global__ void sub_complex128_tensor_kernel(const complex128 *x, const complex128 *y, complex128 *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx].real = x[idx].real - y[idx].real;
    z[idx].imag = x[idx].imag - y[idx].imag;
  }
}

// CUDA kernel for complex64 multiplication
__global__ void mul_complex128_tensor_kernel(const complex128 *x, const complex128 *y, complex128 *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx].real = x[idx].real * y[idx].real - x[idx].imag * y[idx].imag;
    z[idx].imag = x[idx].real * y[idx].imag - x[idx].imag * y[idx].real;
  }
}

// CUDA kernel for left shift op on Tensor
template<typename T>
__global__ void lshift_tensor_kernel(const T *x, const T *y, T *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = x[idx] << y[idx];
  }
}

// CUDA kernel for right shift op on Tensor
template<typename T>
__global__ void rshift_tensor_kernel(const T *x, const T *y, T *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = x[idx] >> y[idx];
  }
}

// CUDA kernel for and op on Tensor
template<typename T>
__global__ void and_tensor_kernel(const T *x, const T *y, T *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = x[idx] & y[idx];
  }
}

// CUDA kernel for nand op on Tensor
template<typename T>
__global__ void nand_tensor_kernel(const T *x, const T *y, T *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = ~(x[idx] & y[idx]);
  }
}

// CUDA kernel for or op on Tensor
template<typename T>
__global__ void or_tensor_kernel(const T *x, const T *y, T *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = x[idx] | y[idx];
  }
}

// CUDA kernel for nor op on Tensor
template<typename T>
__global__ void nor_tensor_kernel(const T *x, const T *y, T *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = ~(x[idx] | y[idx]);
  }
}

// CUDA kernel for not op on Tensor
template<typename T>
__global__ void not_tensor_kernel(const T *x, T *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = ~x[idx];
  }
}

// CUDA kernel for xor op on Tensor
template<typename T>
__global__ void xor_tensor_kernel(const T *x, const T *y, T *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = x[idx] ^ y[idx];
  }
}

// CUDA kernel for xnor op on Tensor
template<typename T>
__global__ void xnor_tensor_kernel(const T *x, const T *y, T *z, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    z[idx] = ~(x[idx] ^ y[idx]);
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
    case CMPX64: add_complex64_tensor_kernel<<<gridSize, blockSize>>>((complex64 *)tx->buf, (complex64 *)ty->buf, (complex64 *)tz->buf, length); break;
    case CMPX128: add_complex128_tensor_kernel<<<gridSize, blockSize>>>((complex128 *)tx->buf, (complex128 *)ty->buf, (complex128 *)tz->buf, length); break;
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
    case CMPX64: sub_complex64_tensor_kernel<<<gridSize, blockSize>>>((complex64 *)tx->buf, (complex64 *)ty->buf, (complex64 *)tz->buf, length); break;
    case CMPX128: sub_complex128_tensor_kernel<<<gridSize, blockSize>>>((complex128 *)tx->buf, (complex128 *)ty->buf, (complex128 *)tz->buf, length); break;
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
    case BOOL: mul_bool_tensor_kernel<<<gridSize, blockSize>>>((bool *)tx->buf, (bool *)ty->buf, (bool *)tz->buf, length); break;
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
    case CMPX64: mul_complex64_tensor_kernel<<<gridSize, blockSize>>>((complex64 *)tx->buf, (complex64 *)ty->buf, (complex64 *)tz->buf, length); break;
    case CMPX128: mul_complex128_tensor_kernel<<<gridSize, blockSize>>>((complex128 *)tx->buf, (complex128 *)ty->buf, (complex128 *)tz->buf, length); break;
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

static PyObject *__tdiv_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  int blockSize = 256;
  int gridSize = (length + blockSize - 1) / blockSize;
  switch(dtype){
    case INT8: tdiv_tensor_kernel<int8, float32><<<gridSize, blockSize>>>((int8 *)tx->buf, (int8 *)ty->buf, (float32 *)tz->buf, length); break;
    case UINT8: tdiv_tensor_kernel<uint8, float32><<<gridSize, blockSize>>>((uint8 *)tx->buf, (uint8 *)ty->buf, (float32 *)tz->buf, length); break;
    case INT16: tdiv_tensor_kernel<int16, float32><<<gridSize, blockSize>>>((int16 *)tx->buf, (int16 *)ty->buf, (float32 *)tz->buf, length); break;
    case UINT16: tdiv_tensor_kernel<uint16, float32><<<gridSize, blockSize>>>((uint16 *)tx->buf, (uint16 *)ty->buf, (float32 *)tz->buf, length); break;
    case INT32: tdiv_tensor_kernel<int32, float32><<<gridSize, blockSize>>>((int32 *)tx->buf, (int32 *)ty->buf, (float32 *)tz->buf, length); break;
    case UINT32: tdiv_tensor_kernel<uint32, float32><<<gridSize, blockSize>>>((uint32 *)tx->buf, (uint32 *)ty->buf, (float32 *)tz->buf, length); break;
    case INT64: tdiv_tensor_kernel<int64, float64><<<gridSize, blockSize>>>((int64 *)tx->buf, (int64 *)ty->buf, (float64 *)tz->buf, length); break;
    case UINT64: tdiv_tensor_kernel<uint64, float64><<<gridSize, blockSize>>>((uint64 *)tx->buf, (uint64 *)ty->buf, (float64 *)tz->buf, length); break;
    case FP32: tdiv_tensor_kernel<float32, float32><<<gridSize, blockSize>>>((float32 *)tx->buf, (float32 *)ty->buf, (float32 *)tz->buf, length); break;
    case FP64: tdiv_tensor_kernel<float64, float64><<<gridSize, blockSize>>>((float64 *)tx->buf, (float64 *)ty->buf, (float64 *)tz->buf, length); break;
    case CMPX64: tdiv_cmpx64_kernel<<<gridSize, blockSize>>>((complex64 *)tx->buf, (complex64 *)ty->buf, (complex64 *)tz->buf, length); break;
    case CMPX128: tdiv_cmpx128_kernel<<<gridSize, blockSize>>>((complex128 *)tx->buf, (complex128 *)ty->buf, (complex128 *)tz->buf, length); break;
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

static PyObject *__fdiv_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  int blockSize = 256;
  int gridSize = (length + blockSize - 1) / blockSize;
  switch(dtype){
    case INT8: fdiv_tensor_kernel<int8, int8><<<gridSize, blockSize>>>((int8 *)tx->buf, (int8 *)ty->buf, (int8 *)tz->buf, length); break;
    case UINT8: fdiv_tensor_kernel<uint8, uint8><<<gridSize, blockSize>>>((uint8 *)tx->buf, (uint8 *)ty->buf, (uint8 *)tz->buf, length); break;
    case INT16: fdiv_tensor_kernel<int16, int16><<<gridSize, blockSize>>>((int16 *)tx->buf, (int16 *)ty->buf, (int16 *)tz->buf, length); break;
    case UINT16: fdiv_tensor_kernel<uint16, uint16><<<gridSize, blockSize>>>((uint16 *)tx->buf, (uint16 *)ty->buf, (uint16 *)tz->buf, length); break;
    case INT32: fdiv_tensor_kernel<int32, int32><<<gridSize, blockSize>>>((int32 *)tx->buf, (int32 *)ty->buf, (int32 *)tz->buf, length); break;
    case UINT32: fdiv_tensor_kernel<uint32, uint32><<<gridSize, blockSize>>>((uint32 *)tx->buf, (uint32 *)ty->buf, (uint32 *)tz->buf, length); break;
    case INT64: fdiv_tensor_kernel<int64, int64><<<gridSize, blockSize>>>((int64 *)tx->buf, (int64 *)ty->buf, (int64 *)tz->buf, length); break;
    case UINT64: fdiv_tensor_kernel<uint64, uint64><<<gridSize, blockSize>>>((uint64 *)tx->buf, (uint64 *)ty->buf, (uint64 *)tz->buf, length); break;
    case FP32: fdiv_tensor_kernel<float32, int64><<<gridSize, blockSize>>>((float32 *)tx->buf, (float32 *)ty->buf, (int64 *)tz->buf, length); break;
    case FP64: fdiv_tensor_kernel<float64, int64><<<gridSize, blockSize>>>((float64 *)tx->buf, (float64 *)ty->buf, (int64 *)tz->buf, length); break;
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

static PyObject *__pow_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype; // assumes dtype uniformity
  int blockSize = 256;
  int gridSize = (length + blockSize - 1) / blockSize;
  switch(dtype){
    case INT8: pow_tensor_kernel<int8><<<gridSize, blockSize>>>((int8 *)tx->buf, (int8 *)ty->buf, (int8 *)tz->buf, length); break;
    case UINT8: pow_tensor_kernel<uint8><<<gridSize, blockSize>>>((uint8 *)tx->buf, (uint8 *)ty->buf, (uint8 *)tz->buf, length); break;
    case INT16: pow_tensor_kernel<int16><<<gridSize, blockSize>>>((int16 *)tx->buf, (int16 *)ty->buf, (int16 *)tz->buf, length); break;
    case UINT16: pow_tensor_kernel<uint16><<<gridSize, blockSize>>>((uint16 *)tx->buf, (uint16 *)ty->buf, (uint16 *)tz->buf, length); break;
    case INT32: pow_tensor_kernel<int32><<<gridSize, blockSize>>>((int32 *)tx->buf, (int32 *)ty->buf, (int32 *)tz->buf, length); break;
    case UINT32: pow_tensor_kernel<uint32><<<gridSize, blockSize>>>((uint32 *)tx->buf, (uint32 *)ty->buf, (uint32 *)tz->buf, length); break;
    case INT64: pow_tensor_kernel<int64><<<gridSize, blockSize>>>((int64 *)tx->buf, (int64 *)ty->buf, (int64 *)tz->buf, length); break;
    case UINT64: pow_tensor_kernel<uint64><<<gridSize, blockSize>>>((uint64 *)tx->buf, (uint64 *)ty->buf, (uint64 *)tz->buf, length); break;
    case FP32: pow_tensor_kernel<float32><<<gridSize, blockSize>>>((float32 *)tx->buf, (float32 *)ty->buf, (float32 *)tz->buf, length); break;
    case FP64: pow_tensor_kernel<float64><<<gridSize, blockSize>>>((float64 *)tx->buf, (float64 *)ty->buf, (float64 *)tz->buf, length); break;
    case CMPX64: pow_cmpx64_tensor_kernel<<<gridSize, blockSize>>>((complex64 *)tx->buf, (complex64 *)ty->buf, (complex64 *)tz->buf, length); break;
    case CMPX128: pow_cmpx128_tensor_kernel<<<gridSize, blockSize>>>((complex128 *)tx->buf, (complex128 *)ty->buf, (complex128 *)tz->buf, length); break;
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

static PyObject *__mod_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype; // assumes dtype uniformity
  int blockSize = 256;
  int gridSize = (length + blockSize - 1) / blockSize;
  switch(dtype){
    case INT8: mod_tensor_kernel<int8><<<gridSize, blockSize>>>((int8 *)tx->buf, (int8 *)ty->buf, (int8 *)tz->buf, length); break;
    case UINT8: mod_tensor_kernel<uint8><<<gridSize, blockSize>>>((uint8 *)tx->buf, (uint8 *)ty->buf, (uint8 *)tz->buf, length); break;
    case INT16: mod_tensor_kernel<int16><<<gridSize, blockSize>>>((int16 *)tx->buf, (int16 *)ty->buf, (int16 *)tz->buf, length); break;
    case UINT16: mod_tensor_kernel<uint16><<<gridSize, blockSize>>>((uint16 *)tx->buf, (uint16 *)ty->buf, (uint16 *)tz->buf, length); break;
    case INT32: mod_tensor_kernel<int32><<<gridSize, blockSize>>>((int32 *)tx->buf, (int32 *)ty->buf, (int32 *)tz->buf, length); break;
    case UINT32: mod_tensor_kernel<uint32><<<gridSize, blockSize>>>((uint32 *)tx->buf, (uint32 *)ty->buf, (uint32 *)tz->buf, length); break;
    case INT64: mod_tensor_kernel<int64><<<gridSize, blockSize>>>((int64 *)tx->buf, (int64 *)ty->buf, (int64 *)tz->buf, length); break;
    case UINT64: mod_tensor_kernel<uint64><<<gridSize, blockSize>>>((uint64 *)tx->buf, (uint64 *)ty->buf, (uint64 *)tz->buf, length); break;
    case FP32: mod_tensor_kernel<float32><<<gridSize, blockSize>>>((float32 *)tx->buf, (float32 *)ty->buf, (float32 *)tz->buf, length); break;
    case FP64: mod_tensor_kernel<float64><<<gridSize, blockSize>>>((float64 *)tx->buf, (float64 *)ty->buf, (float64 *)tz->buf, length); break;
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
  int gridSize = (length + blockSize - 1) / blockSize;
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
  int gridSize = (length + blockSize - 1) / blockSize;
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
  int gridSize = (length + blockSize - 1) / blockSize;
  switch(dtype){
    case INT8: abs_tensor_kernel<int8><<<gridSize, blockSize>>>((int8 *)tx->buf, (int8 *)tz->buf, length); break;
    case UINT8: abs_utensor_kernel<uint8><<<gridSize, blockSize>>>((uint8 *)tx->buf, (uint8 *)tz->buf, length); break;
    case INT16: abs_tensor_kernel<int16><<<gridSize, blockSize>>>((int16 *)tx->buf, (int16 *)tz->buf, length); break;
    case UINT16: abs_utensor_kernel<uint16><<<gridSize, blockSize>>>((uint16 *)tx->buf, (uint16 *)tz->buf, length); break;
    case INT32: abs_tensor_kernel<int32><<<gridSize, blockSize>>>((int32 *)tx->buf, (int32 *)tz->buf, length); break;
    case UINT32: abs_utensor_kernel<uint32><<<gridSize, blockSize>>>((uint32 *)tx->buf, (uint32 *)tz->buf, length); break;
    case INT64: abs_tensor_kernel<int64><<<gridSize, blockSize>>>((int64 *)tx->buf, (int64 *)tz->buf, length); break;
    case UINT64: abs_utensor_kernel<uint64><<<gridSize, blockSize>>>((uint64 *)tx->buf, (uint64 *)tz->buf, length); break;
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


static PyObject *__lshift_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  int blockSize = 256;
  int gridSize = (length + blockSize - 1) / blockSize;
  switch(dtype){
    case INT8: lshift_tensor_kernel<int8><<<gridSize, blockSize>>>((int8 *)tx->buf, (int8 *)ty->buf, (int8 *)tz->buf, length); break;
    case UINT8: lshift_tensor_kernel<uint8><<<gridSize, blockSize>>>((uint8 *)tx->buf, (uint8 *)ty->buf, (uint8 *)tz->buf, length); break;
    case INT16: lshift_tensor_kernel<int16><<<gridSize, blockSize>>>((int16 *)tx->buf, (int16 *)ty->buf, (int16 *)tz->buf, length); break;
    case UINT16: lshift_tensor_kernel<uint16><<<gridSize, blockSize>>>((uint16 *)tx->buf, (uint16 *)ty->buf, (uint16 *)tz->buf, length); break;
    case INT32: lshift_tensor_kernel<int32><<<gridSize, blockSize>>>((int32 *)tx->buf, (int32 *)ty->buf, (int32 *)tz->buf, length); break;
    case UINT32: lshift_tensor_kernel<uint32><<<gridSize, blockSize>>>((uint32 *)tx->buf, (uint32 *)ty->buf, (uint32 *)tz->buf, length); break;
    case INT64: lshift_tensor_kernel<int64><<<gridSize, blockSize>>>((int64 *)tx->buf, (int64 *)ty->buf, (int64 *)tz->buf, length); break;
    case UINT64: lshift_tensor_kernel<uint64><<<gridSize, blockSize>>>((uint64 *)tx->buf, (uint64 *)ty->buf, (uint64 *)tz->buf, length); break;
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

static PyObject *__rshift_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  int blockSize = 256;
  int gridSize = (length + blockSize - 1) / blockSize;
  switch(dtype){
    case INT8: rshift_tensor_kernel<int8><<<gridSize, blockSize>>>((int8 *)tx->buf, (int8 *)ty->buf, (int8 *)tz->buf, length); break;
    case UINT8: rshift_tensor_kernel<uint8><<<gridSize, blockSize>>>((uint8 *)tx->buf, (uint8 *)ty->buf, (uint8 *)tz->buf, length); break;
    case INT16: rshift_tensor_kernel<int16><<<gridSize, blockSize>>>((int16 *)tx->buf, (int16 *)ty->buf, (int16 *)tz->buf, length); break;
    case UINT16: rshift_tensor_kernel<uint16><<<gridSize, blockSize>>>((uint16 *)tx->buf, (uint16 *)ty->buf, (uint16 *)tz->buf, length); break;
    case INT32: rshift_tensor_kernel<int32><<<gridSize, blockSize>>>((int32 *)tx->buf, (int32 *)ty->buf, (int32 *)tz->buf, length); break;
    case UINT32: rshift_tensor_kernel<uint32><<<gridSize, blockSize>>>((uint32 *)tx->buf, (uint32 *)ty->buf, (uint32 *)tz->buf, length); break;
    case INT64: rshift_tensor_kernel<int64><<<gridSize, blockSize>>>((int64 *)tx->buf, (int64 *)ty->buf, (int64 *)tz->buf, length); break;
    case UINT64: rshift_tensor_kernel<uint64><<<gridSize, blockSize>>>((uint64 *)tx->buf, (uint64 *)ty->buf, (uint64 *)tz->buf, length); break;
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

static PyObject *__and_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  int blockSize = 256;
  int gridSize = (length + blockSize - 1) / blockSize;
  switch(dtype){
    case INT8: and_tensor_kernel<int8><<<gridSize, blockSize>>>((int8 *)tx->buf, (int8 *)ty->buf, (int8 *)tz->buf, length); break;
    case UINT8: and_tensor_kernel<uint8><<<gridSize, blockSize>>>((uint8 *)tx->buf, (uint8 *)ty->buf, (uint8 *)tz->buf, length); break;
    case INT16: and_tensor_kernel<int16><<<gridSize, blockSize>>>((int16 *)tx->buf, (int16 *)ty->buf, (int16 *)tz->buf, length); break;
    case UINT16: and_tensor_kernel<uint16><<<gridSize, blockSize>>>((uint16 *)tx->buf, (uint16 *)ty->buf, (uint16 *)tz->buf, length); break;
    case INT32: and_tensor_kernel<int32><<<gridSize, blockSize>>>((int32 *)tx->buf, (int32 *)ty->buf, (int32 *)tz->buf, length); break;
    case UINT32: and_tensor_kernel<uint32><<<gridSize, blockSize>>>((uint32 *)tx->buf, (uint32 *)ty->buf, (uint32 *)tz->buf, length); break;
    case INT64: and_tensor_kernel<int64><<<gridSize, blockSize>>>((int64 *)tx->buf, (int64 *)ty->buf, (int64 *)tz->buf, length); break;
    case UINT64: and_tensor_kernel<uint64><<<gridSize, blockSize>>>((uint64 *)tx->buf, (uint64 *)ty->buf, (uint64 *)tz->buf, length); break;
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

static PyObject *__nand_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  int blockSize = 256;
  int gridSize = (length + blockSize - 1) / blockSize;
  switch(dtype){
    case INT8: nand_tensor_kernel<int8><<<gridSize, blockSize>>>((int8 *)tx->buf, (int8 *)ty->buf, (int8 *)tz->buf, length); break;
    case UINT8: nand_tensor_kernel<uint8><<<gridSize, blockSize>>>((uint8 *)tx->buf, (uint8 *)ty->buf, (uint8 *)tz->buf, length); break;
    case INT16: nand_tensor_kernel<int16><<<gridSize, blockSize>>>((int16 *)tx->buf, (int16 *)ty->buf, (int16 *)tz->buf, length); break;
    case UINT16: nand_tensor_kernel<uint16><<<gridSize, blockSize>>>((uint16 *)tx->buf, (uint16 *)ty->buf, (uint16 *)tz->buf, length); break;
    case INT32: nand_tensor_kernel<int32><<<gridSize, blockSize>>>((int32 *)tx->buf, (int32 *)ty->buf, (int32 *)tz->buf, length); break;
    case UINT32: nand_tensor_kernel<uint32><<<gridSize, blockSize>>>((uint32 *)tx->buf, (uint32 *)ty->buf, (uint32 *)tz->buf, length); break;
    case INT64: nand_tensor_kernel<int64><<<gridSize, blockSize>>>((int64 *)tx->buf, (int64 *)ty->buf, (int64 *)tz->buf, length); break;
    case UINT64: nand_tensor_kernel<uint64><<<gridSize, blockSize>>>((uint64 *)tx->buf, (uint64 *)ty->buf, (uint64 *)tz->buf, length); break;
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

static PyObject *__or_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  int blockSize = 256;
  int gridSize = (length + blockSize - 1) / blockSize;
  switch(dtype){
    case INT8: or_tensor_kernel<int8><<<gridSize, blockSize>>>((int8 *)tx->buf, (int8 *)ty->buf, (int8 *)tz->buf, length); break;
    case UINT8: or_tensor_kernel<uint8><<<gridSize, blockSize>>>((uint8 *)tx->buf, (uint8 *)ty->buf, (uint8 *)tz->buf, length); break;
    case INT16: or_tensor_kernel<int16><<<gridSize, blockSize>>>((int16 *)tx->buf, (int16 *)ty->buf, (int16 *)tz->buf, length); break;
    case UINT16: or_tensor_kernel<uint16><<<gridSize, blockSize>>>((uint16 *)tx->buf, (uint16 *)ty->buf, (uint16 *)tz->buf, length); break;
    case INT32: or_tensor_kernel<int32><<<gridSize, blockSize>>>((int32 *)tx->buf, (int32 *)ty->buf, (int32 *)tz->buf, length); break;
    case UINT32: or_tensor_kernel<uint32><<<gridSize, blockSize>>>((uint32 *)tx->buf, (uint32 *)ty->buf, (uint32 *)tz->buf, length); break;
    case INT64: or_tensor_kernel<int64><<<gridSize, blockSize>>>((int64 *)tx->buf, (int64 *)ty->buf, (int64 *)tz->buf, length); break;
    case UINT64: or_tensor_kernel<uint64><<<gridSize, blockSize>>>((uint64 *)tx->buf, (uint64 *)ty->buf, (uint64 *)tz->buf, length); break;
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

static PyObject *__nor_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  int blockSize = 256;
  int gridSize = (length + blockSize - 1) / blockSize;
  switch(dtype){
    case INT8: nor_tensor_kernel<int8><<<gridSize, blockSize>>>((int8 *)tx->buf, (int8 *)ty->buf, (int8 *)tz->buf, length); break;
    case UINT8: nor_tensor_kernel<uint8><<<gridSize, blockSize>>>((uint8 *)tx->buf, (uint8 *)ty->buf, (uint8 *)tz->buf, length); break;
    case INT16: nor_tensor_kernel<int16><<<gridSize, blockSize>>>((int16 *)tx->buf, (int16 *)ty->buf, (int16 *)tz->buf, length); break;
    case UINT16: nor_tensor_kernel<uint16><<<gridSize, blockSize>>>((uint16 *)tx->buf, (uint16 *)ty->buf, (uint16 *)tz->buf, length); break;
    case INT32: nor_tensor_kernel<int32><<<gridSize, blockSize>>>((int32 *)tx->buf, (int32 *)ty->buf, (int32 *)tz->buf, length); break;
    case UINT32: nor_tensor_kernel<uint32><<<gridSize, blockSize>>>((uint32 *)tx->buf, (uint32 *)ty->buf, (uint32 *)tz->buf, length); break;
    case INT64: nor_tensor_kernel<int64><<<gridSize, blockSize>>>((int64 *)tx->buf, (int64 *)ty->buf, (int64 *)tz->buf, length); break;
    case UINT64: nor_tensor_kernel<uint64><<<gridSize, blockSize>>>((uint64 *)tx->buf, (uint64 *)ty->buf, (uint64 *)tz->buf, length); break;
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

static PyObject *__not_tensor__(const tensor_t *tx, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  int blockSize = 256;
  int gridSize = (length + blockSize - 1) / blockSize;
  switch(dtype){
    case INT8: not_tensor_kernel<int8><<<gridSize, blockSize>>>((int8 *)tx->buf, (int8 *)tz->buf, length); break;
    case UINT8: not_tensor_kernel<uint8><<<gridSize, blockSize>>>((uint8 *)tx->buf, (uint8 *)tz->buf, length); break;
    case INT16: not_tensor_kernel<int16><<<gridSize, blockSize>>>((int16 *)tx->buf, (int16 *)tz->buf, length); break;
    case UINT16: not_tensor_kernel<uint16><<<gridSize, blockSize>>>((uint16 *)tx->buf, (uint16 *)tz->buf, length); break;
    case INT32: not_tensor_kernel<int32><<<gridSize, blockSize>>>((int32 *)tx->buf, (int32 *)tz->buf, length); break;
    case UINT32: not_tensor_kernel<uint32><<<gridSize, blockSize>>>((uint32 *)tx->buf, (uint32 *)tz->buf, length); break;
    case INT64: not_tensor_kernel<int64><<<gridSize, blockSize>>>((int64 *)tx->buf, (int64 *)tz->buf, length); break;
    case UINT64: not_tensor_kernel<uint64><<<gridSize, blockSize>>>((uint64 *)tx->buf, (uint64 *)tz->buf, length); break;
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

static PyObject *__xor_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  int blockSize = 256;
  int gridSize = (length + blockSize - 1) / blockSize;
  switch(dtype){
    case INT8: xor_tensor_kernel<int8><<<gridSize, blockSize>>>((int8 *)tx->buf, (int8 *)ty->buf, (int8 *)tz->buf, length); break;
    case UINT8: xor_tensor_kernel<uint8><<<gridSize, blockSize>>>((uint8 *)tx->buf, (uint8 *)ty->buf, (uint8 *)tz->buf, length); break;
    case INT16: xor_tensor_kernel<int16><<<gridSize, blockSize>>>((int16 *)tx->buf, (int16 *)ty->buf, (int16 *)tz->buf, length); break;
    case UINT16: xor_tensor_kernel<uint16><<<gridSize, blockSize>>>((uint16 *)tx->buf, (uint16 *)ty->buf, (uint16 *)tz->buf, length); break;
    case INT32: xor_tensor_kernel<int32><<<gridSize, blockSize>>>((int32 *)tx->buf, (int32 *)ty->buf, (int32 *)tz->buf, length); break;
    case UINT32: xor_tensor_kernel<uint32><<<gridSize, blockSize>>>((uint32 *)tx->buf, (uint32 *)ty->buf, (uint32 *)tz->buf, length); break;
    case INT64: xor_tensor_kernel<int64><<<gridSize, blockSize>>>((int64 *)tx->buf, (int64 *)ty->buf, (int64 *)tz->buf, length); break;
    case UINT64: xor_tensor_kernel<uint64><<<gridSize, blockSize>>>((uint64 *)tx->buf, (uint64 *)ty->buf, (uint64 *)tz->buf, length); break;
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

static PyObject *__xnor_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  int blockSize = 256;
  int gridSize = (length + blockSize - 1) / blockSize;
  switch(dtype){
    case INT8: xnor_tensor_kernel<int8><<<gridSize, blockSize>>>((int8 *)tx->buf, (int8 *)ty->buf, (int8 *)tz->buf, length); break;
    case UINT8: xnor_tensor_kernel<uint8><<<gridSize, blockSize>>>((uint8 *)tx->buf, (uint8 *)ty->buf, (uint8 *)tz->buf, length); break;
    case INT16: xnor_tensor_kernel<int16><<<gridSize, blockSize>>>((int16 *)tx->buf, (int16 *)ty->buf, (int16 *)tz->buf, length); break;
    case UINT16: xnor_tensor_kernel<uint16><<<gridSize, blockSize>>>((uint16 *)tx->buf, (uint16 *)ty->buf, (uint16 *)tz->buf, length); break;
    case INT32: xnor_tensor_kernel<int32><<<gridSize, blockSize>>>((int32 *)tx->buf, (int32 *)ty->buf, (int32 *)tz->buf, length); break;
    case UINT32: xnor_tensor_kernel<uint32><<<gridSize, blockSize>>>((uint32 *)tx->buf, (uint32 *)ty->buf, (uint32 *)tz->buf, length); break;
    case INT64: xnor_tensor_kernel<int64><<<gridSize, blockSize>>>((int64 *)tx->buf, (int64 *)ty->buf, (int64 *)tz->buf, length); break;
    case UINT64: xnor_tensor_kernel<uint64><<<gridSize, blockSize>>>((uint64 *)tx->buf, (uint64 *)ty->buf, (uint64 *)tz->buf, length); break;
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

static PyObject *tdiv(PyObject *self, PyObject *args){
  PyObject *x, *y;
  if(!PyArg_ParseTuple(args, "OO", &x, &y))
    return NULL;
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
  if(tx->dtype == FP64)
    tz->dtype = FP64;
  else if(tx->dtype == CMPX64)
    tz->dtype = CMPX64;
  else if(tx->dtype == CMPX128)
    tz->dtype = CMPX128;
  else
    tz->dtype = FP32;   // integers + FP32 → FP32 output
  tz->size = tx->size;
  tz->ndim = tx->ndim;
  tz->device = (device_t){CUDA, 0};
  tz->stride = NULL;
  switch(tz->dtype){
    case FP32:   tz->element_size = sizeof(float32);   break;
    case FP64:   tz->element_size = sizeof(float64);   break;
    case CMPX64: tz->element_size = sizeof(complex64); break;
    case CMPX128:tz->element_size = sizeof(complex128);break;
    default:
      free(tz);
      PyErr_SetString(PyExc_RuntimeError,
                      "Unsupported dtype in tdiv");
      return NULL;
  }
  if(tx->ndim > 0 && tx->shape){
    tz->shape = (size_t *)malloc(sizeof(size_t) * tz->ndim);
    if(!tz->shape){
      free(tz);
      PyErr_SetString(PyExc_RuntimeError, "tensor_t.shape allocation failed!");
      return NULL;
    }
    for(size_t i = 0; i < tz->ndim; i++)
      tz->shape[i] = tx->shape[i];
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
    PyErr_Format(PyExc_RuntimeError, "cudaMalloc failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  tz->buf = tz->storage->ptr;
  return __tdiv_tensor__(tx, ty, tz);
}
static PyObject *fdiv(PyObject *self, PyObject *args){
  PyObject *x, *y;
  if(!PyArg_ParseTuple(args, "OO", &x, &y)) return NULL;
  if(!PyCapsule_CheckExact(x) || !PyCapsule_CheckExact(y)){
    PyErr_SetString(PyExc_RuntimeError, "operands must be tensor_t capsules");
    return NULL;
  }
  tensor_t *tx = (tensor_t *)PyCapsule_GetPointer(x, "tensor_t on CUDA");
  tensor_t *ty = (tensor_t *)PyCapsule_GetPointer(y, "tensor_t on CUDA");
  if(!tx || !ty){
    PyErr_SetString(PyExc_RuntimeError,
      "invalid tensor capsule pointer");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_ValueError, "tensors must have the same size for floordiv");
    return NULL;
  }
  if(tx->dtype != ty->dtype){
    PyErr_SetString(PyExc_TypeError, "tensors must have the same dtype for floordiv");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "'//' operation on complex tensor is not supported");
    return NULL;
  }
  tensor_t *tz = alloc_result_tensor(tx);
  if(!tz){
    PyErr_SetString(PyExc_MemoryError,
      "failed to allocate output tensor");
    return NULL;
  }
  return __fdiv_tensor__(tx, ty, tz);
}

static PyObject *pow_(PyObject *self, PyObject *args){
  PyObject *x, *y;
  if(!PyArg_ParseTuple(args, "OO", &x, &y)) return NULL;
  if(!PyCapsule_CheckExact(x) || !PyCapsule_CheckExact(y)){
    PyErr_SetString(PyExc_RuntimeError, "operands must be tensor_t capsules");
    return NULL;
  }
  tensor_t *tx = (tensor_t *)PyCapsule_GetPointer(x, "tensor_t on CUDA");
  tensor_t *ty = (tensor_t *)PyCapsule_GetPointer(y, "tensor_t on CUDA");
  if(!tx || !ty){
    PyErr_SetString(PyExc_RuntimeError,
      "invalid tensor capsule pointer");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_ValueError, "tensors must have the same size for floordiv");
    return NULL;
  }
  if(tx->dtype != ty->dtype){
    PyErr_SetString(PyExc_TypeError, "tensors must have the same dtype for floordiv");
    return NULL;
  }
  tensor_t *tz = alloc_result_tensor(tx);
  if(!tz){
    PyErr_SetString(PyExc_MemoryError,
      "failed to allocate output tensor");
    return NULL;
  }
  return __pow_tensor__(tx, ty, tz);
}

static PyObject *mod_(PyObject *self, PyObject *args){
  PyObject *x, *y;
  if(!PyArg_ParseTuple(args, "OO", &x, &y)) return NULL;
  if(!PyCapsule_CheckExact(x) || !PyCapsule_CheckExact(y)){
    PyErr_SetString(PyExc_RuntimeError, "operands must be tensor_t capsules");
    return NULL;
  }
  tensor_t *tx = (tensor_t *)PyCapsule_GetPointer(x, "tensor_t on CUDA");
  tensor_t *ty = (tensor_t *)PyCapsule_GetPointer(y, "tensor_t on CUDA");
  if(!tx || !ty){
    PyErr_SetString(PyExc_RuntimeError,
      "invalid tensor capsule pointer");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_ValueError, "tensors must have the same size for floordiv");
    return NULL;
  }
  if(tx->dtype != ty->dtype){
    PyErr_SetString(PyExc_TypeError, "tensors must have the same dtype for floordiv");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "%% operation on complex tensor is not supported");
    return NULL;
  }
  tensor_t *tz = alloc_result_tensor(tx);
  if(!tz){
    PyErr_SetString(PyExc_MemoryError,
      "failed to allocate output tensor");
    return NULL;
  }
  return __mod_tensor__(tx, ty, tz);
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
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "> operation on complex tensor is not supported");
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
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, ">= operation on complex tensor is not supported");
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
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "< operation on complex tensor is not supported");
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
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "<= operation on complex tensor is not supported");
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

static PyObject *lshift(PyObject *self, PyObject *args){
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
  if(tx->dtype == FP32 || tx->dtype == FP64){
    PyErr_SetString(PyExc_RuntimeError, "'<<' operation on floating points tensor is not supported");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "'<<' operation on complex tensor is not supported");
    return NULL;
  }
  if(tx->device.type != ty->device.type || tx->device.type != CUDA){
    PyErr_SetString(PyExc_RuntimeError, "Both tensor_t(s) must be on same device (CUDA)");
    return NULL;
  }
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __lshift_tensor__(tx, ty, tz);
}

static PyObject *rshift(PyObject *self, PyObject *args){
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
  if(tx->dtype == FP32 || tx->dtype == FP64){
    PyErr_SetString(PyExc_RuntimeError, "'<<' operation on floating points tensor is not supported");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "'<<' operation on complex tensor is not supported");
    return NULL;
  }
  if(tx->device.type != ty->device.type || tx->device.type != CUDA){
    PyErr_SetString(PyExc_RuntimeError, "Both tensor_t(s) must be on same device (CUDA)");
    return NULL;
  }
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __rshift_tensor__(tx, ty, tz);
}

static PyObject *and_(PyObject *self, PyObject *args){
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
  if(tx->dtype == FP32 || tx->dtype == FP64){
    PyErr_SetString(PyExc_RuntimeError, "'<<' operation on floating points tensor is not supported");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "'<<' operation on complex tensor is not supported");
    return NULL;
  }
  if(tx->device.type != ty->device.type || tx->device.type != CUDA){
    PyErr_SetString(PyExc_RuntimeError, "Both tensor_t(s) must be on same device (CUDA)");
    return NULL;
  }
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __and_tensor__(tx, ty, tz);
}

static PyObject *nand_(PyObject *self, PyObject *args){
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
  if(tx->dtype == FP32 || tx->dtype == FP64){
    PyErr_SetString(PyExc_RuntimeError, "'<<' operation on floating points tensor is not supported");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "'<<' operation on complex tensor is not supported");
    return NULL;
  }
  if(tx->device.type != ty->device.type || tx->device.type != CUDA){
    PyErr_SetString(PyExc_RuntimeError, "Both tensor_t(s) must be on same device (CUDA)");
    return NULL;
  }
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __nand_tensor__(tx, ty, tz);
}

static PyObject *or_(PyObject *self, PyObject *args){
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
  if(tx->dtype == FP32 || tx->dtype == FP64){
    PyErr_SetString(PyExc_RuntimeError, "'<<' operation on floating points tensor is not supported");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "'<<' operation on complex tensor is not supported");
    return NULL;
  }
  if(tx->device.type != ty->device.type || tx->device.type != CUDA){
    PyErr_SetString(PyExc_RuntimeError, "Both tensor_t(s) must be on same device (CUDA)");
    return NULL;
  }
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __or_tensor__(tx, ty, tz);
}

static PyObject *nor_(PyObject *self, PyObject *args){
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
  if(tx->dtype == FP32 || tx->dtype == FP64){
    PyErr_SetString(PyExc_RuntimeError, "'<<' operation on floating points tensor is not supported");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "'<<' operation on complex tensor is not supported");
    return NULL;
  }
  if(tx->device.type != ty->device.type || tx->device.type != CUDA){
    PyErr_SetString(PyExc_RuntimeError, "Both tensor_t(s) must be on same device (CUDA)");
    return NULL;
  }
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __nor_tensor__(tx, ty, tz);
}

static PyObject *not_(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "operands must be the tensor capsule");
    return NULL;
  }
  tensor_t *tx = (tensor_t *)PyCapsule_GetPointer(x, "tensor_t on CUDA");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  if(tx->dtype == FP32 || tx->dtype == FP64){
    PyErr_SetString(PyExc_RuntimeError, "'<<' operation on floating points tensor is not supported");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "'<<' operation on complex tensor is not supported");
    return NULL;
  }
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __not_tensor__(tx, tz);
}

static PyObject *xor_(PyObject *self, PyObject *args){
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
  if(tx->dtype == FP32 || tx->dtype == FP64){
    PyErr_SetString(PyExc_RuntimeError, "'<<' operation on floating points tensor is not supported");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "'<<' operation on complex tensor is not supported");
    return NULL;
  }
  if(tx->device.type != ty->device.type || tx->device.type != CUDA){
    PyErr_SetString(PyExc_RuntimeError, "Both tensor_t(s) must be on same device (CUDA)");
    return NULL;
  }
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __xor_tensor__(tx, ty, tz);
}

static PyObject *xnor_(PyObject *self, PyObject *args){
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
  if(tx->dtype == FP32 || tx->dtype == FP64){
    PyErr_SetString(PyExc_RuntimeError, "'<<' operation on floating points tensor is not supported");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "'<<' operation on complex tensor is not supported");
    return NULL;
  }
  if(tx->device.type != ty->device.type || tx->device.type != CUDA){
    PyErr_SetString(PyExc_RuntimeError, "Both tensor_t(s) must be on same device (CUDA)");
    return NULL;
  }
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __xnor_tensor__(tx, ty, tz);
}

__global__ void permute_kernel(
  const char *in_buf,
  char *out_buf,
  const size_t *in_stride,
  const size_t *out_shape,
  const int *axes,
  int ndim,
  size_t total,
  size_t itemsize
){
  size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  if(linear >= total) return;
  size_t tmp = linear;
  size_t in_offset = 0;
  for(int d = ndim - 1; d >= 0; d--){
    size_t idx = tmp % out_shape[d];
    tmp /= out_shape[d];
    int orig_axis = axes[d];
    in_offset += idx * in_stride[orig_axis];
  }
  const char *src = in_buf + in_offset * itemsize;
  char *dst = out_buf + linear * itemsize;
  for(size_t b = 0; b < itemsize; b++){
    dst[b] = src[b];
  }
}

static PyObject *permute(PyObject *self, PyObject *args){
    PyObject *x;
    PyObject *axes_tuple;
    if(!PyArg_ParseTuple(args, "OO", &x, &axes_tuple)) return NULL;
    if(!PyCapsule_CheckExact(x)){
      PyErr_SetString(PyExc_TypeError, "expected tensor capsule");
      return NULL;
    }
    tensor_t *t = (tensor_t *)PyCapsule_GetPointer(x, "tensor_t on CUDA");
    if(!t){
      PyErr_SetString(PyExc_RuntimeError, "invalid tensor capsule");
      return NULL;
    }
    if(!PyTuple_Check(axes_tuple)){
      PyErr_SetString(PyExc_TypeError, "permute expects tuple");
      return NULL;
    }
    int ndim = t->ndim;
    if(PyTuple_Size(axes_tuple) != ndim){
      PyErr_SetString(PyExc_ValueError, "axes must match ndim");
      return NULL;
    }
    int *axes_host = (int *)malloc(sizeof(int) * ndim);
    bool *used = (bool *)calloc(ndim, sizeof(bool));
    for(int i = 0; i < ndim; i++){
      int ax = (int)PyLong_AsLong(PyTuple_GetItem(axes_tuple, i));
      if(ax < 0) ax += ndim;
      if(ax < 0 || ax >= ndim){
        free(axes_host);
        free(used);
        PyErr_SetString(PyExc_IndexError, "axis out of range");
        return NULL;
      }
      if(used[ax]){
        free(axes_host);
        free(used);
        PyErr_SetString(PyExc_ValueError, "duplicate axis");
        return NULL;
      }
      used[ax] = true;
      axes_host[i] = ax;
    }
    free(used);
    tensor_t *out = (tensor_t *)malloc(sizeof(tensor_t));
    out->dtype = t->dtype;
    out->device = (device_t){CUDA, t->device.index};
    out->ndim = ndim;
    out->element_size = t->element_size;
    out->size = t->size;
    out->shape = (size_t *)malloc(sizeof(size_t) * ndim);
    out->stride = (size_t *)malloc(sizeof(size_t) * ndim);
    for(int i = 0; i < ndim; i++){
      out->shape[i] = t->shape[axes_host[i]];
    }
    out->stride[ndim - 1] = 1;
    for(int i = ndim - 2; i >= 0; i--){
      out->stride[i] = out->stride[i + 1] * out->shape[i + 1];
    }
    size_t itemsize = getsize(out->dtype);
    out->storage = (storage_t *)malloc(sizeof(storage_t));
    out->storage->refcount = 1;
    out->storage->device = out->device;
    out->storage->bytes = out->size * itemsize;
    cudaMalloc(&out->storage->ptr, out->storage->bytes);
    out->buf = out->storage->ptr;
    int *axes_dev;
    size_t *in_stride_dev;
    size_t *out_shape_dev;
    cudaMalloc(&axes_dev, sizeof(int) * ndim);
    cudaMalloc(&in_stride_dev, sizeof(size_t) * ndim);
    cudaMalloc(&out_shape_dev, sizeof(size_t) * ndim);
    cudaMemcpy(axes_dev, axes_host, sizeof(int) * ndim, cudaMemcpyHostToDevice);
    cudaMemcpy(in_stride_dev, t->stride, sizeof(size_t) * ndim, cudaMemcpyHostToDevice);
    cudaMemcpy(out_shape_dev, out->shape, sizeof(size_t) * ndim, cudaMemcpyHostToDevice);
    free(axes_host);
    size_t total = out->size;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    permute_kernel<<<blocks, threads>>>(
      (char *)t->buf,
      (char *)out->buf,
      in_stride_dev,
      out_shape_dev,
      axes_dev,
      ndim,
      total,
      itemsize
    );
    cudaDeviceSynchronize();
    cudaFree(axes_dev);
    cudaFree(in_stride_dev);
    cudaFree(out_shape_dev);
    return PyCapsule_New(out, "tensor_t on CUDA", capsule_destroyer);
}

tensor_t *tensor_empty_like_bmm(tensor_t *tx, size_t m, size_t n){
  tensor_t *tz = (tensor_t *)malloc(sizeof(tensor_t));
  if(!tz) return NULL;
  tz->dtype = tx->dtype;
  tz->device = tx->device;
  tz->ndim = tx->ndim;
  tz->element_size = tx->element_size;
  tz->shape = (size_t *)malloc(sizeof(size_t) * tz->ndim);
  if(!tz->shape){ free(tz); return NULL; }
  for(size_t i = 0; i < tz->ndim - 2; i++)
    tz->shape[i] = tx->shape[i];
  tz->shape[tz->ndim - 2] = m;
  tz->shape[tz->ndim - 1] = n;
  tz->stride = (size_t *)malloc(sizeof(size_t) * tz->ndim);
  if(!tz->stride){ free(tz->shape); free(tz); return NULL; }
  tz->stride[tz->ndim - 1] = 1;
  for(int i = tz->ndim - 2; i >= 0; i--)
    tz->stride[i] = tz->stride[i+1] * tz->shape[i+1];
  tz->size = 1;
  for(size_t i = 0; i < tz->ndim; i++)
    tz->size *= tz->shape[i];
  tz->storage = (storage_t *)malloc(sizeof(storage_t));
  if(!tz->storage){
    free(tz->stride);
    free(tz->shape);
    free(tz);
    return NULL;
  }
  tz->storage->bytes = tz->size * tz->element_size;
  tz->storage->refcount = 1;
  tz->storage->device = tz->device;
  CUDA_CHECK(cudaMalloc(&tz->storage->ptr, tz->storage->bytes));
  tz->buf = tz->storage->ptr;
  CUDA_CHECK(cudaMemset(tz->buf, 0, tz->storage->bytes));
  return tz;
}

template<typename T>
__global__ void bmm_tensor_kernel(const T *x, const T *y, T *z, size_t m, size_t k, size_t n){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t batch = blockIdx.y;
  size_t total = m * n;
  if(idx >= total) return;
  size_t i = idx / n;
  size_t j = idx % n;
  size_t a_base = batch * m * k;
  size_t b_base = batch * k * n;
  size_t c_base = batch * m * n;
  T sum = (T)0;
  for(size_t kk = 0; kk < k; kk++){
    sum += x[a_base + i * k + kk] * y[b_base + kk * n + j];
  }
  z[c_base + i * n + j] = sum;
}

template<typename C, typename R>
__global__ void bmm_cmpx_tensor_kernel(const C *x, const C *y, C *z, size_t m, size_t k, size_t n){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t batch = blockIdx.y;
  size_t total = m * n;
  if(idx >= total) return;
  size_t i = idx / n;
  size_t j = idx % n;
  size_t a_base = batch * m * k;
  size_t b_base = batch * k * n;
  size_t c_base = batch * m * n;
  R sum_real = (R)0;
  R sum_imag = (R)0;
  for(size_t kk = 0; kk < k; kk++){
    C a = x[a_base + i * k + kk];
    C b = y[b_base + kk * n + j];
    sum_real += a.real * b.real - a.imag * b.imag;
    sum_imag += a.real * b.imag + a.imag * b.real;
  }
  z[c_base + i * n + j].real = sum_real;
  z[c_base + i * n + j].imag = sum_imag;
}

static PyObject *bmm(PyObject *self, PyObject *args){
  PyObject *x, *y;
  if(!PyArg_ParseTuple(args, "OO", &x, &y)) return NULL;
  if(!PyCapsule_CheckExact(x) || !PyCapsule_CheckExact(y)){
    PyErr_SetString(PyExc_RuntimeError, "operands must be tensor capsule");
    return NULL;
  }
  tensor_t *tx = (tensor_t *)PyCapsule_GetPointer(x, "tensor_t on CUDA");
  tensor_t *ty = (tensor_t *)PyCapsule_GetPointer(y, "tensor_t on CUDA");
  if(!tx || !ty){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  if(tx->dtype != ty->dtype){
    PyErr_SetString(PyExc_TypeError, "Both tensors must have same dtype");
    return NULL;
  }
  if(tx->device.type != CUDA || ty->device.type != CUDA){
    PyErr_SetString(PyExc_RuntimeError, "Both tensors must be on CUDA");
    return NULL;
  }
  if(tx->ndim < 3 || ty->ndim < 3){
    PyErr_SetString(PyExc_RuntimeError, "bmm requires tensors >= 3D");
    return NULL;
  }
  if(tx->ndim != ty->ndim){
    PyErr_SetString(PyExc_ValueError, "Both tensors must have same ndim");
    return NULL;
  }
  size_t m = tx->shape[tx->ndim - 2];
  size_t k = tx->shape[tx->ndim - 1];
  size_t k2 = ty->shape[ty->ndim - 2];
  size_t n = ty->shape[ty->ndim - 1];
  if(k != k2){
    PyErr_SetString(PyExc_ValueError, "Matrix dimensions incompatible");
    return NULL;
  }
  for(size_t i = 0; i < tx->ndim - 2; i++){
    if(tx->shape[i] != ty->shape[i]){
      PyErr_SetString(PyExc_ValueError, "Batch dimensions must match");
      return NULL;
    }
  }
  tensor_t *tz = tensor_empty_like_bmm(tx, m, n);
  if(!tz){
    PyErr_SetString(PyExc_MemoryError, "Output allocation failed");
    return NULL;
  }
  size_t batch_size = 1;
  for(size_t i = 0; i < tx->ndim - 2; i++){
    batch_size *= tx->shape[i];
  }
  dim3 block(256);
  dim3 grid((m * n + block.x - 2) / block.x, batch_size);
  switch (tx->dtype){
    case INT8: {
      int8 *x = (int8 *)tx->buf;
      int8 *y = (int8 *)ty->buf;
      int8 *z = (int8 *)tz->buf;
      bmm_tensor_kernel<int8><<<grid, block>>>(x, y, z, m , k, n);
      cudaDeviceSynchronize();
      break;
    }
    case UINT8: {
      uint8 *x = (uint8 *)tx->buf;
      uint8 *y = (uint8 *)ty->buf;
      uint8 *z = (uint8 *)tz->buf;
      bmm_tensor_kernel<uint8><<<grid, block>>>(x, y, z, m , k, n);
      cudaDeviceSynchronize();
      break;
    }
    case INT16: {
      int16 *x = (int16 *)tx->buf;
      int16 *y = (int16 *)ty->buf;
      int16 *z = (int16 *)tz->buf;
      bmm_tensor_kernel<int16><<<grid, block>>>(x, y, z, m, k, n);
      cudaDeviceSynchronize();
      break;
    }
    case UINT16: {
      uint16 *x = (uint16 *)tx->buf;
      uint16 *y = (uint16 *)ty->buf;
      uint16 *z = (uint16 *)tz->buf;
      bmm_tensor_kernel<uint16><<<grid, block>>>(x, y, z, m, k, n);
      cudaDeviceSynchronize();
      break;
    }
    case INT32: {
      int32 *x = (int32 *)tx->buf;
      int32 *y = (int32 *)ty->buf;
      int32 *z = (int32 *)tz->buf;
      bmm_tensor_kernel<int32><<<grid, block>>>(x, y, z, m, k, n);
      cudaDeviceSynchronize();
      break;
    }
    case UINT32: {
      uint32 *x = (uint32 *)tx->buf;
      uint32 *y = (uint32 *)ty->buf;
      uint32 *z = (uint32 *)tz->buf;
      bmm_tensor_kernel<uint32><<<grid, block>>>(x, y, z, m, k, n);
      cudaDeviceSynchronize();
      break;}
    case INT64: {
      int64 *x = (int64 *)tx->buf;
      int64 *y = (int64 *)ty->buf;
      int64 *z = (int64 *)tz->buf;
      bmm_tensor_kernel<int64><<<grid, block>>>(x, y, z, m, k, n);
      cudaDeviceSynchronize();
      break;
    }
    case UINT64: {
      int64 *x = (int64 *)tx->buf;
      int64 *y = (int64 *)ty->buf;
      int64 *z = (int64 *)tz->buf;
      bmm_tensor_kernel<int64><<<grid, block>>>(x, y, z, m, k, n);
      cudaDeviceSynchronize();
      break;
    }
    case FP32: {
      float32 *x = (float32 *)tx->buf;
      float32 *y = (float32 *)ty->buf;
      float32 *z = (float32 *)tz->buf;
      bmm_tensor_kernel<float32><<<grid, block>>>(x, y, z, m, k, n);
      cudaDeviceSynchronize();
      break;
    }
    case FP64: {
      float64 *x = (float64 *)tx->buf;
      float64 *y = (float64 *)ty->buf;
      float64 *z = (float64 *)tz->buf;
      bmm_tensor_kernel<float64><<<grid, block>>>(x, y, z, m, k, n);
      cudaDeviceSynchronize();
      break;
    }
    case CMPX64: {
      complex64 *x = (complex64 *)tx->buf;
      complex64 *y = (complex64 *)ty->buf;
      complex64 *z = (complex64 *)tz->buf;
      bmm_cmpx_tensor_kernel<complex64, float><<<grid, block>>>(x, y, z, m, k, n);
      cudaDeviceSynchronize();
      break;
    }
    case CMPX128: {
      complex128 *x = (complex128 *)tx->buf;
      complex128 *y = (complex128 *)ty->buf;
      complex128 *z = (complex128 *)tz->buf;
      bmm_cmpx_tensor_kernel<complex128, double><<<grid, block>>>(x, y, z, m, k, n);
      cudaDeviceSynchronize();
      break;
    }
    default: {
      PyErr_SetString(PyExc_RuntimeError, "something bad happened at `bmm` function");
      return NULL;
    }
  }
  return PyCapsule_New(tz, "tensor_t on CUDA", capsule_destroyer);
}

tensor_t *tensor_empty_axis_sum(tensor_t *tx, int axis){
  tensor_t *tz = (tensor_t *)malloc(sizeof(tensor_t));
  if(!tz) return NULL;
  tz->dtype = tx->dtype;
  tz->device = tx->device;
  tz->element_size = tx->element_size;
  tz->ndim = tx->ndim - 1;
  tz->shape = (size_t *)malloc(sizeof(size_t) * tz->ndim);
  tz->stride = (size_t *)malloc(sizeof(size_t) * tz->ndim);
  if(!tz->shape || !tz->stride){
    free(tz->shape);
    free(tz->stride);
    free(tz);
    return NULL;
  }
  int j = 0;
  for(int i = 0; i < tx->ndim; i++){
    if(i == axis) continue;
    tz->shape[j++] = tx->shape[i];
  }
  tz->stride[tz->ndim - 1] = 1;
  for(int i = tz->ndim - 2; i >= 0; i--){
    tz->stride[i] = tz->stride[i + 1] * tz->shape[i + 1];
  }
  tz->size = 1;
  for(int i = 0; i < tz->ndim; i++){
    tz->size *= tz->shape[i];
  }
  tz->storage = (storage_t *)malloc(sizeof(storage_t));
  if(!tz->storage){
    free(tz->shape);
    free(tz->stride);
    free(tz);
    return NULL;
  }
  tz->storage->bytes = tz->size * tz->element_size;
  tz->storage->refcount = 1;
  tz->storage->device = tz->device;
  CUDA_CHECK(cudaMalloc(&tz->storage->ptr, tz->storage->bytes));
  tz->buf = tz->storage->ptr;
  CUDA_CHECK(cudaMemset(tz->buf, 0, tz->storage->bytes));
  return tz;
}

tensor_t *tensor_empty_scalar_like(tensor_t *tx){
  tensor_t *tz = (tensor_t *)malloc(sizeof(tensor_t));
  if(!tz) return NULL;
  tz->dtype = tx->dtype;
  tz->device = tx->device;
  tz->element_size = tx->element_size;
  tz->ndim = 0;
  tz->shape = NULL;
  tz->stride = NULL;
  tz->size = 1;
  tz->storage = (storage_t *)malloc(sizeof(storage_t));
  if(!tz->storage){
    free(tz);
    return NULL;
  }
  tz->storage->bytes = tz->element_size;
  tz->storage->refcount = 1;
  tz->storage->device = tz->device;
  CUDA_CHECK(cudaMalloc(&tz->storage->ptr, tz->storage->bytes));
  tz->buf = tz->storage->ptr;
  CUDA_CHECK(cudaMemset(tz->buf, 0, tz->storage->bytes));
  return tz;
}

__device__ inline uint64 atomicAdd_u64(uint64 *addr, uint64 val){
  unsigned long long *uaddr = (unsigned long long *)addr;
  unsigned long long old = *uaddr, assumed;
  do{
    assumed = old;
    old = atomicCAS(uaddr, assumed, assumed + (unsigned long long)val);
  }while(assumed != old);
  return (uint64)old;
}

__device__ inline int64 atomicAdd_i64(int64 *addr, int64 val){
  unsigned long long *uaddr = (unsigned long long *)addr;
  unsigned long long old = *uaddr, assumed;
  do{
    assumed = old;
    old = atomicCAS(
      uaddr,
      assumed,
      (unsigned long long)((long long)assumed + (long long)val)
    );
  }while(assumed != old);
  return (int64)old;
}

__device__ inline double atomicAdd_f64(double *addr, double val){
#if __CUDA_ARCH__ >= 600
  return atomicAdd(addr, val);
#else
  unsigned long long *uaddr = (unsigned long long *)addr;
  unsigned long long old = *uaddr, assumed;
  do{
    assumed = old;
    old = atomicCAS(
      uaddr,
      assumed,
      __double_as_longlong(val + __longlong_as_double(assumed))
    );
  }while(assumed != old);
  return __longlong_as_double(old);
#endif
}

#define SUM_ALL_KERNEL(NAME, IN_T, ACC_T, ATOMIC_FN)  \
__global__ void sum_all_##NAME##_kernel(              \
    const IN_T *x, ACC_T *out, size_t N               \
){                                                    \
  __shared__ ACC_T cache[256];                        \
  size_t tid = threadIdx.x;                           \
  size_t idx = blockIdx.x * blockDim.x + tid;         \
  ACC_T temp = 0;                                     \
  while(idx < N){                                     \
    temp += (ACC_T)x[idx];                            \
    idx += blockDim.x * gridDim.x;                    \
  }                                                   \
  cache[tid] = temp;                                  \
  __syncthreads();                                    \
  for(size_t s = blockDim.x/2; s > 0; s >>= 1){       \
    if(tid < s) cache[tid] += cache[tid+s];           \
    __syncthreads();                                  \
  }                                                   \
  if(tid == 0) ATOMIC_FN(out, cache[0]);              \
}

#define SUM_AXIS_KERNEL(NAME, IN_T, ACC_T)                             \
__global__ void sum_axis_##NAME##_kernel(                              \
    const IN_T *in, IN_T *out,                                         \
    const size_t *stride, const size_t *out_stride,                    \
    int ndim, int axis, size_t reduced_dim, size_t out_size            \
){                                                                     \
  size_t out_index = blockIdx.x * blockDim.x + threadIdx.x; \
  if(out_index >= out_size) return;                         \
  int idx_in[16] = {0};                                     \
  size_t tmp = out_index;                                   \
  int j = 0;                                                \
  for(int i = 0; i < ndim; i++){                            \
    if(i == axis) continue;                                 \
    idx_in[i] = tmp / out_stride[j];                        \
    tmp %= out_stride[j];                                   \
    j++;                                                    \
  }                                                         \
  ACC_T total = 0;                                          \
  for(size_t r = 0; r < reduced_dim; r++){                  \
    idx_in[axis] = r;                                       \
    size_t offset = 0;                                      \
    for(int k = 0; k < ndim; k++)                           \
      offset += idx_in[k] * stride[k];                      \
    total += (ACC_T)in[offset];                             \
  }                                                         \
  out[out_index] = (IN_T)total;                             \
}

SUM_ALL_KERNEL(int8, int8, int32, atomicAdd)
SUM_ALL_KERNEL(uint8, uint8, uint64, atomicAdd_u64)
SUM_ALL_KERNEL(int16, int16, int32, atomicAdd)
SUM_ALL_KERNEL(uint16, uint16, uint64, atomicAdd_u64)
SUM_ALL_KERNEL(int32, int32, int64, atomicAdd_i64)
SUM_ALL_KERNEL(uint32, uint32, uint64, atomicAdd_u64)
SUM_ALL_KERNEL(int64, int64, int64, atomicAdd_i64)
SUM_ALL_KERNEL(uint64, uint64, uint64, atomicAdd_u64)
SUM_ALL_KERNEL(fp32, float32, float32, atomicAdd)
SUM_ALL_KERNEL(fp64, float64, float64, atomicAdd_f64)

SUM_AXIS_KERNEL(int8, int8, int32)
SUM_AXIS_KERNEL(uint8, uint8, uint64)
SUM_AXIS_KERNEL(int16, int16, int32)
SUM_AXIS_KERNEL(uint16, uint16, uint64)
SUM_AXIS_KERNEL(int32, int32, int64)
SUM_AXIS_KERNEL(uint32, uint32, uint64)
SUM_AXIS_KERNEL(int64, int64, int64)
SUM_AXIS_KERNEL(uint64, uint64, uint64)
SUM_AXIS_KERNEL(fp32, float32, float32)
SUM_AXIS_KERNEL(fp64, float64, float64)

__global__ void sum_axis_cmpx64_kernel(
  const complex64 *in, complex64 *out,
  const size_t *stride, const size_t *out_stride,
  int ndim, int axis, size_t reduced_dim, size_t out_size
){
  size_t out_index = blockIdx.x * blockDim.x + threadIdx.x;
  if(out_index >= out_size) return;
  int idx_in[16] = {0};
  size_t tmp = out_index;
  int j = 0;
  for(int i = 0; i < ndim; i++){
    if(i == axis) continue;
    idx_in[i] = tmp / out_stride[j];
    tmp %= out_stride[j];
    j++;
  }
  float real_sum = 0.0f;
  float imag_sum = 0.0f;
  for(size_t r = 0; r < reduced_dim; r++){
    idx_in[axis] = r;
    size_t offset = 0;
    for(int k = 0; k < ndim; k++)
      offset += idx_in[k] * stride[k];
    real_sum += in[offset].real;
    imag_sum += in[offset].imag;
  }
  out[out_index].real = real_sum;
  out[out_index].imag = imag_sum;
}

__global__ void sum_axis_cmpx128_kernel(
  const complex128 *in, complex128 *out,
  const size_t *stride, const size_t *out_stride,
  int ndim, int axis, size_t reduced_dim, size_t out_size
){
  size_t out_index = blockIdx.x * blockDim.x + threadIdx.x;
  if(out_index >= out_size) return;
  int idx_in[16] = {0};
  size_t tmp = out_index;
  int j = 0;
  for(int i = 0; i < ndim; i++){
    if(i == axis) continue;
    idx_in[i] = tmp / out_stride[j];
    tmp %= out_stride[j];
    j++;
  }
  double real_sum = 0.0;
  double imag_sum = 0.0;
  for(size_t r = 0; r < reduced_dim; r++){
    idx_in[axis] = r;
    size_t offset = 0;
    for(int k = 0; k < ndim; k++)
      offset += idx_in[k] * stride[k];
    real_sum += in[offset].real;
    imag_sum += in[offset].imag;
  }
  out[out_index].real = real_sum;
  out[out_index].imag = imag_sum;
}

__global__ void sum_all_cmpx64_kernel(const complex64 *x, complex64 *out, size_t N){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= N) return;
  atomicAdd(&out->real, x[idx].real);
  atomicAdd(&out->imag, x[idx].imag);
}

__global__ void sum_all_cmpx128_kernel(const complex128 *x, complex128 *out, size_t N){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= N) return;
  atomicAdd_f64(&out->real, x[idx].real);
  atomicAdd_f64(&out->imag, x[idx].imag);
}

static PyObject *__sum_all__(tensor_t *tx, tensor_t *tz){
  size_t N = tx->size;
  CUDA_CHECK(cudaMemset(tz->buf, 0, tz->storage->bytes));
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  if(blocks > 1024) blocks = 1024;
  switch(tx->dtype){
    case INT8: sum_all_int8_kernel<<<blocks,threads>>>((int8*)tx->buf,(int32*)tz->buf,N); break;
    case UINT8: sum_all_uint8_kernel<<<blocks,threads>>>((uint8*)tx->buf,(uint64*)tz->buf,N); break;
    case INT16: sum_all_int16_kernel<<<blocks,threads>>>((int16*)tx->buf,(int32*)tz->buf,N); break;
    case UINT16: sum_all_uint16_kernel<<<blocks,threads>>>((uint16*)tx->buf,(uint64*)tz->buf,N); break;
    case INT32: sum_all_int32_kernel<<<blocks,threads>>>((int32*)tx->buf,(int64*)tz->buf,N); break;
    case UINT32: sum_all_uint32_kernel<<<blocks,threads>>>((uint32*)tx->buf,(uint64*)tz->buf,N); break;
    case INT64: sum_all_int64_kernel<<<blocks,threads>>>((int64*)tx->buf,(int64*)tz->buf,N); break;
    case UINT64: sum_all_uint64_kernel<<<blocks,threads>>>((uint64*)tx->buf,(uint64*)tz->buf,N); break;
    case FP32: sum_all_fp32_kernel<<<blocks,threads>>>((float32*)tx->buf,(float32*)tz->buf,N); break;
    case FP64: sum_all_fp64_kernel<<<blocks,threads>>>((float64*)tx->buf,(float64*)tz->buf,N); break;
    case CMPX64: sum_all_cmpx64_kernel<<<blocks,threads>>>((complex64*)tx->buf,(complex64*)tz->buf,N); break;
    case CMPX128: sum_all_cmpx128_kernel<<<blocks,threads>>>((complex128*)tx->buf,(complex128*)tz->buf,N); break;
    default:
      destroy(tz);
      PyErr_SetString(PyExc_TypeError,"sum not supported on CUDA");
      return NULL;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  return PyCapsule_New(tz,"tensor_t on CUDA",capsule_destroyer);
}

static PyObject *__sum_axis__(tensor_t *tx, tensor_t *tz, int axis){
  size_t reduced_dim = tx->shape[axis];
  size_t out_size = tz->size;
  size_t *d_stride, *d_out_stride;
  CUDA_CHECK(cudaMalloc(&d_stride, sizeof(size_t) * tx->ndim));
  CUDA_CHECK(cudaMalloc(&d_out_stride, sizeof(size_t) * tz->ndim));
  CUDA_CHECK(cudaMemcpy(d_stride, tx->stride, sizeof(size_t) * tx->ndim, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_out_stride, tz->stride, sizeof(size_t) * tz->ndim, cudaMemcpyHostToDevice));
  int threads = 256;
  int blocks = (out_size + threads - 1) / threads;
  switch(tx->dtype){
    case INT8:
      sum_axis_int8_kernel<<<blocks,threads>>>(
        (int8*)tx->buf,(int8*)tz->buf,
        d_stride,d_out_stride,
        tx->ndim,axis,reduced_dim,out_size);
      break;
    case UINT8:
      sum_axis_uint8_kernel<<<blocks,threads>>>(
        (uint8*)tx->buf,(uint8*)tz->buf,
        d_stride,d_out_stride,
        tx->ndim,axis,reduced_dim,out_size);
      break;
    case INT16:
      sum_axis_int16_kernel<<<blocks,threads>>>(
        (int16*)tx->buf,(int16*)tz->buf,
        d_stride,d_out_stride,
        tx->ndim,axis,reduced_dim,out_size);
      break;
    case UINT16:
      sum_axis_uint16_kernel<<<blocks,threads>>>(
        (uint16*)tx->buf,(uint16*)tz->buf,
        d_stride,d_out_stride,
        tx->ndim,axis,reduced_dim,out_size);
      break;
    case INT32:
      sum_axis_int32_kernel<<<blocks,threads>>>(
        (int32*)tx->buf,(int32*)tz->buf,
        d_stride,d_out_stride,
        tx->ndim,axis,reduced_dim,out_size);
      break;
    case UINT32:
      sum_axis_uint32_kernel<<<blocks,threads>>>(
        (uint32*)tx->buf,(uint32*)tz->buf,
        d_stride,d_out_stride,
        tx->ndim,axis,reduced_dim,out_size);
      break;
    case INT64:
      sum_axis_int64_kernel<<<blocks,threads>>>(
        (int64*)tx->buf,(int64*)tz->buf,
        d_stride,d_out_stride,
        tx->ndim,axis,reduced_dim,out_size);
      break;
    case UINT64:
      sum_axis_uint64_kernel<<<blocks,threads>>>(
        (uint64*)tx->buf,(uint64*)tz->buf,
        d_stride,d_out_stride,
        tx->ndim,axis,reduced_dim,out_size);
      break;
    case FP32:
      sum_axis_fp32_kernel<<<blocks,threads>>>(
        (float32*)tx->buf,(float32*)tz->buf,
        d_stride,d_out_stride,
        tx->ndim,axis,reduced_dim,out_size);
      break;
    case FP64:
      sum_axis_fp64_kernel<<<blocks,threads>>>(
        (float64*)tx->buf,(float64*)tz->buf,
        d_stride,d_out_stride,
        tx->ndim,axis,reduced_dim,out_size);
      break;
    case CMPX64:
      sum_axis_cmpx64_kernel<<<blocks,threads>>>(
        (complex64*)tx->buf,(complex64*)tz->buf,
        d_stride,d_out_stride,
        tx->ndim,axis,reduced_dim,out_size);
      break;
    case CMPX128:
      sum_axis_cmpx128_kernel<<<blocks,threads>>>(
        (complex128*)tx->buf,(complex128*)tz->buf,
        d_stride,d_out_stride,
        tx->ndim,axis,reduced_dim,out_size);
      break;
    default:
      destroy(tz);
      PyErr_SetString(PyExc_TypeError, "sum(axis) not supported on CUDA");
      return NULL;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaFree(d_stride);
  cudaFree(d_out_stride);
  return PyCapsule_New(tz, "tensor_t on CUDA", capsule_destroyer);
}

static PyObject *sum(PyObject *self, PyObject *args, PyObject *kwargs){
  PyObject *x;
  PyObject *axis_obj = Py_None;
  static char *kwlist[] = {(char *)"x", (char *)"axis", NULL};
  if(!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist, &x, &axis_obj))
    return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid capsule");
    return NULL;
  }
  tensor_t *tx = (tensor_t *)PyCapsule_GetPointer(x, "tensor_t on CUDA");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor pointer");
    return NULL;
  }
  if(tx->device.type != CUDA){
    PyErr_SetString(PyExc_RuntimeError, "Tensor must be on CUDA");
    return NULL;
  }
  if(axis_obj == Py_None){
    tensor_t *tz = tensor_empty_scalar_like(tx);
    if(!tz){
      PyErr_SetString(PyExc_RuntimeError, "scalar tensor allocation failed");
      return NULL;
    }
    return __sum_all__(tx, tz);
  }
  if(!PyLong_Check(axis_obj)){
    PyErr_SetString(PyExc_TypeError, "axis must be int or None");
    return NULL;
  }
  int axis = PyLong_AsLong(axis_obj);
  if(axis < 0) axis += tx->ndim;
  if(axis < 0 || axis >= tx->ndim){
    PyErr_SetString(PyExc_ValueError, "axis out of range");
    return NULL;
  }
  if(tx->ndim == 1){
    tensor_t *tz = tensor_empty_scalar_like(tx);
    if(!tz){
      PyErr_SetString(PyExc_RuntimeError, "scalar tensor allocation failed");
      return NULL;
    }
    return __sum_all__(tx, tz);
  }
  tensor_t *tz = tensor_empty_axis_sum(tx, axis);
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "axis-sum tensor allocation failed");
    return NULL;
  }
  return __sum_axis__(tx, tz, axis);
}

static tensor_t *tensor_empty_like_realimag(tensor_t *tx, dtype_t out_dtype){
  tensor_t *tz = (tensor_t *)malloc(sizeof(tensor_t));
  if(!tz) return NULL;
  tz->dtype = out_dtype;
  tz->element_size = getsize(out_dtype);
  tz->device = tx->device;
  tz->ndim = tx->ndim;
  tz->shape = (size_t *)malloc(sizeof(size_t) * tz->ndim);
  tz->stride = (size_t *)malloc(sizeof(size_t) * tz->ndim);
  if(!tz->shape || !tz->stride){
    free(tz->shape);
    free(tz->stride);
    free(tz);
    return NULL;
  }
  for(size_t i = 0; i < tz->ndim; i++){
    tz->shape[i] = tx->shape[i];
    tz->stride[i] = tx->stride[i];
  }
  tz->size = tx->size;
  tz->storage = (storage_t *)malloc(sizeof(storage_t));
  if(!tz->storage){
    free(tz->shape);
    free(tz->stride);
    free(tz);
    return NULL;
  }
  tz->storage->device = tz->device;
  tz->storage->refcount = 1;
  tz->storage->bytes = tz->size * tz->element_size;
  CUDA_CHECK(cudaMalloc(&tz->storage->ptr, tz->storage->bytes));
  tz->buf = tz->storage->ptr;
  return tz;
}

__global__ void real_cmpx64_kernel(const complex64 *x, float32 *out, size_t N){
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < N) out[i] = x[i].real;
}

__global__ void real_cmpx128_kernel(const complex128 *x, float64 *out, size_t N){
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < N) out[i] = x[i].real;
}

__global__ void imag_cmpx64_kernel(const complex64 *x, float32 *out, size_t N){
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < N) out[i] = x[i].imag;
}

__global__ void imag_cmpx128_kernel(const complex128 *x, float64 *out, size_t N){
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < N) out[i] = x[i].imag;
}

static PyObject *real(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "invalid tensor capsule");
    return NULL;
  }
  tensor_t *tx = (tensor_t *)PyCapsule_GetPointer(x, "tensor_t on CUDA");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  if(tx->dtype != CMPX64 && tx->dtype != CMPX128){
    PyErr_SetString(PyExc_TypeError, "real() implemented only for complex dtype tensor(s)");
    return NULL;
  }
  dtype_t out_dtype = (tx->dtype == CMPX64) ? FP32 : FP64;
  tensor_t *tz;
  if(tx->ndim == 0){
    tz = tensor_empty_scalar_like(tx);
    tz->dtype = out_dtype;
    tz->element_size = getsize(out_dtype);
    tz->storage->bytes = tz->element_size;
    if(tx->dtype == CMPX64){
      float32 tmp = ((complex64*)tx->buf)[0].real;
      CUDA_CHECK(cudaMemcpy(tz->buf, &tmp, sizeof(float32), cudaMemcpyHostToDevice));
    }else{
      float64 tmp = ((complex128*)tx->buf)[0].real;
      CUDA_CHECK(cudaMemcpy(tz->buf, &tmp, sizeof(float64), cudaMemcpyHostToDevice));
    }
    return PyCapsule_New(tz, "tensor_t on CUDA", capsule_destroyer);
  }
  tz = tensor_empty_like_realimag(tx, out_dtype);
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "output allocation failed");
    return NULL;
  }
  size_t N = tx->size;
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  if(tx->dtype == CMPX64){ real_cmpx64_kernel<<<blocks, threads>>>((complex64*)tx->buf, (float32*)tz->buf, N); }
  else{ real_cmpx128_kernel<<<blocks, threads>>>((complex128*)tx->buf, (float64*)tz->buf, N); }
  CUDA_CHECK(cudaDeviceSynchronize());
  return PyCapsule_New(tz, "tensor_t on CUDA", capsule_destroyer);
}

static PyObject *imag(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "invalid tensor capsule");
    return NULL;
  }
  tensor_t *tx = (tensor_t *)PyCapsule_GetPointer(x, "tensor_t on CUDA");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  if(tx->dtype != CMPX64 && tx->dtype != CMPX128){
    PyErr_SetString(PyExc_TypeError, "imag() implemented only for complex dtype tensor(s)");
    return NULL;
  }
  dtype_t out_dtype = (tx->dtype == CMPX64) ? FP32 : FP64;
  tensor_t *tz;
  if(tx->ndim == 0){
    tz = tensor_empty_scalar_like(tx);
    tz->dtype = out_dtype;
    tz->element_size = getsize(out_dtype);
    tz->storage->bytes = tz->element_size;
    if(tx->dtype == CMPX64){
      float32 tmp = ((complex64*)tx->buf)[0].imag;
      CUDA_CHECK(cudaMemcpy(tz->buf, &tmp, sizeof(float32), cudaMemcpyHostToDevice));
    }else{
      float64 tmp = ((complex128*)tx->buf)[0].imag;
      CUDA_CHECK(cudaMemcpy(tz->buf, &tmp, sizeof(float64), cudaMemcpyHostToDevice));
    }
    return PyCapsule_New(tz, "tensor_t on CUDA", capsule_destroyer);
  }
  tz = tensor_empty_like_realimag(tx, out_dtype);
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "output allocation failed");
    return NULL;
  }
  size_t N = tx->size;
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  if(tx->dtype == CMPX64){ imag_cmpx64_kernel<<<blocks, threads>>>((complex64*)tx->buf, (float32*)tz->buf, N); }
  else{ imag_cmpx128_kernel<<<blocks, threads>>>((complex128*)tx->buf, (float64*)tz->buf, N); }
  CUDA_CHECK(cudaDeviceSynchronize());
  return PyCapsule_New(tz, "tensor_t on CUDA", capsule_destroyer);
}

static PyMethodDef methods[] = {
  {"add", add, METH_VARARGS, "element-wise 'add' operation on CUDA tensor"},
  {"sub", sub, METH_VARARGS, "element-wise 'sub' operation on CUDA tensor"},
  {"mul", mul, METH_VARARGS, "element-wise 'mul' operation on CUDA tensor"},
  {"tdiv", tdiv, METH_VARARGS, "element-wise 'tdiv' operation on CUDA tensor"},
  {"fdiv", fdiv, METH_VARARGS, "element-wise 'fdiv' operation on CUDA tensor"},
  {"pow", pow_, METH_VARARGS, "element-wise 'pow' operation on CUDA tensor"},
  {"mod", mod_, METH_VARARGS, "element-wise 'mod' operation on CUDA tensor"},
  {"eq", eq, METH_VARARGS, "element-wise 'eq' operation on CUDA tensor"},
  {"ne", ne, METH_VARARGS, "element-wise 'ne' operation on CUDA tensor"},
  {"gt", gt, METH_VARARGS, "element-wise 'gt' operation on CUDA tensor"},
  {"ge", ge, METH_VARARGS, "element-wise 'ge' operation on CUDA tensor"},
  {"lt", lt, METH_VARARGS, "element-wise 'lt' operation on CUDA tensor"},
  {"le", le, METH_VARARGS, "element-wise 'le' operation on CUDA tensor"},
  {"neg", neg, METH_VARARGS, "element-wise 'neg' operation on CUDA tensor"},
  {"pos", pos, METH_VARARGS, "element-wise 'pos' operation on CUDA tensor"},
  {"abs", abs_, METH_VARARGS, "element-wise 'abs' operation on CUDA tensor"},
  {"lshift", lshift, METH_VARARGS, "element-wise 'lshift' operation on CUDA tensor"},
  {"rshift", rshift, METH_VARARGS, "element-wise 'rshift' operation on CUDA tensor"},
  {"and_", and_, METH_VARARGS, "element-wise 'and' operation on CUDA tensor"},
  {"nand_", nand_, METH_VARARGS, "element-wise 'nand' operation on CUDA tensor"},
  {"or_", or_, METH_VARARGS, "element-wise 'or' operation on CUDA tensor"},
  {"nor_", nor_, METH_VARARGS, "element-wise 'nor' operation on CUDA tensor"},
  {"not_", not_, METH_VARARGS, "element-wise 'not' operation on CUDA tensor"},
  {"xor_", xor_, METH_VARARGS, "element-wise 'xor' operation on CUDA tensor"},
  {"xnor_", xnor_, METH_VARARGS, "element-wise 'xnor' operation on CUDA tensor"},
  {"sum", (PyCFunction)sum, METH_VARARGS | METH_KEYWORDS, "returns the sum of CUDA tensor"},
  {"permute", permute, METH_VARARGS, "permute on CUDA tensor"},
  {"bmm", bmm, METH_VARARGS, "compute batch matrix multiplication on CUDA tensor"},
  {"real", real, METH_VARARGS, "get real values from complex tensor on CUDA"},
  {"imag", imag, METH_VARARGS, "get imag values from complex tensor on CUDA"},
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
