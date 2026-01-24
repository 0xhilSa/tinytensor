#include <cuda_runtime.h>
#include <driver_types.h>
#include <python3.10/Python.h>
#include <python3.10/methodobject.h>
#include "../tensor.h"

#define KERNEL_LAUNCH(kernel, blocks, threads, ...)  \
  do{ \
    kernel<<<blocks, threads>>>(__VA_ARGS__); \
  } while (0)

static void capsule_destructor(PyObject *capsule){
  tensor_t *t = (tensor_t *)PyCapsule_GetPointer(capsule, "tensor_t on CUDA");
  if(!t) return;
  cudaFree(t->data);
  destroy(t);
  free(t);
}

__global__ void add_tensor_int8(const i8 *x, const i8 *y, i8 *out, size_t length){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length) out[idx] = x[idx] + y[idx];
}

void __add_tensor_kernel_dispatch__(const void *x, const void *y, void *out, size_t length, dtype_t dtype){
  int threads = 256;
  int blocks = (length + threads - 1) / threads;
  switch(dtype){
    case INT8: KERNEL_LAUNCH(add_tensor_int8, blocks, threads, (i8 *)x, (i8 *)y, (i8 *)out, length); break;
    default: break;
  }
}
void __add_tensor_scalar_kernel_dispatch__(const void *x, const void *y, void *out, size_t length, dtype_t dtype){}

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
