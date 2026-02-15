#include <python3.10/Python.h>
#include <cuda_runtime.h>
#include "../tensor.h"

#define SQRT_2_DIV_PI 0.7978845608028654

#define CUDA_CHECK_OBJ(err)                                       \
  if((err) != cudaSuccess){                                       \
    PyErr_SetString(PyExc_RuntimeError, cudaGetErrorString(err)); \
    return NULL;                                                  \
  }

void capsule_destroyer(PyObject *capsule){
  tensor_t *t = (tensor_t *)PyCapsule_GetPointer(capsule, "tensor_t on CUDA");
  if(t){
    destroy(t);
  }
}

static tensor_t *alloc_result_tensor(const tensor_t *t, dtype_t out_dtype){
  tensor_t *tz = (tensor_t *)malloc(sizeof(tensor_t));
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
    return NULL;
  }
  tz->dtype = out_dtype;
  tz->size  = t->size;
  tz->ndim  = t->ndim;
  tz->element_size = getsize(out_dtype);
  tz->device = t->device;
  tz->stride = (size_t *)malloc(sizeof(size_t) * tz->ndim);
  if(!tz->stride){
    if(tz->shape) free(tz->shape);
    free(tz);
    PyErr_SetString(PyExc_RuntimeError, "tensor_t.stride allocation failed!");
    return NULL;
  }

  for(size_t i = 0; i < tz->ndim; i++){
    tz->stride[i] = t->stride[i];
  }
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

template<typename T>
__global__ void relu_kernel(const T *x, T *z, size_t length){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){ z[idx] = (x[idx] > (T)0) ? x[idx] : (T)0; }
}

template<typename T>
__global__ void relu_uint_kernel(const T *x, T *z, size_t length){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){ z[idx] = (T)x[idx]; }
}

template<typename Tin, typename Tout>
__global__ void leaky_kernel(const Tin *x, Tout *z, size_t length, Tout alpha){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    Tout val = (Tout)x[idx];
    z[idx] = (val > (Tout)0) ? val : alpha * val;
  }
}

template<typename Tin, typename Tout>
__global__ void gelu_kernel(const Tin *x, Tout *z, size_t length){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    Tout val = (Tout)x[idx];
    Tout cube = val * val * val;
    Tout inner = (Tout)SQRT_2_DIV_PI * (val + (Tout)0.044715 * cube);
    z[idx] = (Tout)0.5 * val * ((Tout)1.0 + tanh(inner));
  }
}

static PyObject *__relu__(const tensor_t *tx, tensor_t *tz){
  dtype_t dtype = tx->dtype;
  size_t length = tx->size;
  int blockSize = 256;
  int gridSize = (length + blockSize - 1) / blockSize;
  switch(dtype){
    case INT8: relu_kernel<int8><<<gridSize, blockSize>>>((int8 *)tx->buf, (int8 *)tz->buf, length); break;
    case UINT8: relu_kernel<uint8><<<gridSize, blockSize>>>((uint8 *)tx->buf, (uint8 *)tz->buf, length); break;
    case INT16: relu_kernel<int16><<<gridSize, blockSize>>>((int16 *)tx->buf, (int16 *)tz->buf, length); break;
    case UINT16: relu_kernel<uint16><<<gridSize, blockSize>>>((uint16 *)tx->buf, (uint16 *)tz->buf, length); break;
    case INT32: relu_kernel<int32><<<gridSize, blockSize>>>((int32 *)tx->buf, (int32 *)tz->buf, length); break;
    case UINT32: relu_kernel<uint32><<<gridSize, blockSize>>>((uint32 *)tx->buf, (uint32 *)tz->buf, length); break;
    case INT64: relu_kernel<int64><<<gridSize, blockSize>>>((int64 *)tx->buf, (int64 *)tz->buf, length); break;
    case UINT64: relu_kernel<uint64><<<gridSize, blockSize>>>((uint64 *)tx->buf, (uint64 *)tz->buf, length); break;
    case FP32: relu_kernel<float32><<<gridSize, blockSize>>>((float32 *)tx->buf, (float32 *)tz->buf, length); break;
    case FP64: relu_kernel<float64><<<gridSize, blockSize>>>((float64 *)tx->buf, (float64 *)tz->buf, length); break;
  }
  CUDA_CHECK_OBJ(cudaDeviceSynchronize());
  return PyCapsule_New(tz, "tensor_t on CUDA", capsule_destroyer);
}

static PyObject *__leaky_relu__(const tensor_t *tx, tensor_t *tz, double negative_slope){
  size_t length = tx->size;
  int blockSize = 256;
  int gridSize  = (length + blockSize - 1) / blockSize;
  switch(tx->dtype){
    case INT8:
    case INT16:
    case INT32: leaky_kernel<int32, float32><<<gridSize, blockSize>>>((int32*)tx->buf, (float32*)tz->buf, length, (float32)negative_slope); break;
    case INT64: leaky_kernel<int64, float64><<<gridSize, blockSize>>>((int64*)tx->buf, (float64*)tz->buf, length, (float64)negative_slope); break;
    case UINT8: relu_uint_kernel<uint8><<<gridSize, blockSize>>>((uint8*)tx->buf, (uint8*)tz->buf, length); break;
    case UINT16: relu_uint_kernel<uint16><<<gridSize, blockSize>>>((uint16*)tx->buf, (uint16*)tz->buf, length); break;
    case UINT32: relu_uint_kernel<uint32><<<gridSize, blockSize>>>((uint32*)tx->buf, (uint32*)tz->buf, length); break;
    case UINT64: relu_uint_kernel<uint64><<<gridSize, blockSize>>>((uint64*)tx->buf, (uint64*)tz->buf, length); break;
    case FP32: leaky_kernel<float32, float32><<<gridSize, blockSize>>>((float32*)tx->buf, (float32*)tz->buf, length, (float32)negative_slope); break;
    case FP64: leaky_kernel<float64, float64><<<gridSize, blockSize>>>((float64*)tx->buf, (float64*)tz->buf, length, (float64)negative_slope); break;
  }
  CUDA_CHECK_OBJ(cudaDeviceSynchronize());
  return PyCapsule_New(tz, "tensor_t on CUDA", capsule_destroyer);
}

static PyObject *__gelu_launch__(const tensor_t *tx, tensor_t *tz){
  size_t length = tx->size;
  int blockSize = 256;
  int gridSize  = (length + blockSize - 1) / blockSize;
  switch(tx->dtype){
    case INT8:
    case INT16:
    case INT32: gelu_kernel<int32, float32><<<gridSize, blockSize>>>((int32*)tx->buf, (float32*)tz->buf, length); break;
    case INT64: gelu_kernel<int64, float64><<<gridSize, blockSize>>>((int64*)tx->buf, (float64*)tz->buf, length); break;
    case UINT8: relu_uint_kernel<uint8><<<gridSize, blockSize>>>((uint8*)tx->buf, (uint8*)tz->buf, length); break;
    case UINT16: relu_uint_kernel<uint16><<<gridSize, blockSize>>>((uint16*)tx->buf, (uint16*)tz->buf, length); break;
    case UINT32: relu_uint_kernel<uint32><<<gridSize, blockSize>>>((uint32*)tx->buf, (uint32*)tz->buf, length); break;
    case UINT64: relu_uint_kernel<uint64><<<gridSize, blockSize>>>((uint64*)tx->buf, (uint64*)tz->buf, length); break;
    case FP32: gelu_kernel<float32, float32><<<gridSize, blockSize>>>((float32*)tx->buf, (float32*)tz->buf, length); break;
    case FP64: gelu_kernel<float64, float64><<<gridSize, blockSize>>>((float64*)tx->buf, (float64*)tz->buf, length); break;
  }
  CUDA_CHECK_OBJ(cudaDeviceSynchronize());
  return PyCapsule_New(tz, "tensor_t on CUDA", capsule_destroyer);
}

static PyObject *relu_act(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid capsule pointer");
    return NULL;
  }
  tensor_t *tx = (tensor_t *)PyCapsule_GetPointer(x, "tensor_t on CUDA");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor pointer");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "relu is not supported for complex dtype tensors");
    return NULL;
  }
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx, tx->dtype);
  return __relu__(tx, tz);
}

static PyObject *leaky_relu_act(PyObject *self, PyObject *args){
  PyObject *x;
  double negative_slope=0.01;
  if(!PyArg_ParseTuple(args, "O|d", &x, &negative_slope)) return NULL;
  tensor_t *tx = (tensor_t *)PyCapsule_GetPointer(x, "tensor_t on CUDA");
  dtype_t out_dtype = tx->dtype;
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "relu is not supported for complex dtype tensors");
    return NULL;
  }
  else if(tx->dtype == INT8 || tx->dtype == INT16 || tx->dtype == INT32) out_dtype = FP32;
  else if(tx->dtype == INT64) out_dtype = FP64;
  tensor_t *tz = alloc_result_tensor(tx, out_dtype);
  if(!tz) return NULL;
  return __leaky_relu__(tx, tz, negative_slope);
}

static PyObject *gelu_act(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid capsule pointer");
    return NULL;
  }
  tensor_t *tx = (tensor_t *)PyCapsule_GetPointer(x, "tensor_t on CUDA");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor pointer");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "gelu not supported for complex dtype tensors");
    return NULL;
  }
  dtype_t out_dtype = tx->dtype;
  if(tx->dtype == INT8 || tx->dtype == INT16 || tx->dtype == INT32)
    out_dtype = FP32;
  else if(tx->dtype == INT64)
     out_dtype = FP64;
  tensor_t *tz = alloc_result_tensor(tx, out_dtype);
  if(!tz) return NULL;
  return __gelu_launch__(tx, tz);
}

static PyMethodDef methods[] = {
  {"relu", relu_act, METH_VARARGS, "element-wsie 'relu' op on CUDA tensor"},
  {"leaky_relu", leaky_relu_act, METH_VARARGS, "CUDA LeakyReLU"},
  {"gelu", gelu_act, METH_VARARGS, "CUDA GELU"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "functional_cuda",
  "CUDA functional ops",
  -1,
  methods
};

PyMODINIT_FUNC PyInit_functional_cuda(void){
  return PyModule_Create(&module);
}

