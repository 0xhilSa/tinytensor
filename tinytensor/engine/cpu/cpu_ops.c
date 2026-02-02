#include <python3.10/Python.h>
#include "../tensor.h"

void capsule_destroyer(PyObject *capsule){
  tensor_t *t = PyCapsule_GetPointer(capsule, "tensor_t on CPU");
  if(t){
    destroy(t);
  }
}

typedef enum {L, R} side_t;

static tensor_t *alloc_result_tensor(const tensor_t *t){
  tensor_t *tz = malloc(sizeof(tensor_t));
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
    tz->shape = malloc(sizeof(size_t) * tz->ndim);
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
  tz->storage = malloc(sizeof(storage_t));
  if(!tz->storage){
    if(tz->shape) free(tz->shape);
    free(tz);
    PyErr_SetString(PyExc_RuntimeError, "tensor_t.storage allocation failed!");
    return NULL;
  }
  tz->storage->bytes = tz->size * tz->element_size;
  tz->storage->device = tz->device;
  tz->storage->refcount = 1;
  tz->storage->ptr = malloc(tz->storage->bytes);
  if(!tz->storage->ptr){
    if(tz->shape) free(tz->shape);
    free(tz->storage);
    free(tz);
    PyErr_SetString(PyExc_RuntimeError, "tensor_t.storage.ptr allocation failed!");
    return NULL;
  }
  tz->buf = tz->storage->ptr;
  return tz;
}

static PyObject *__add_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  char *px = (char *)tx->buf;
  char *py = (char *)ty->buf;
  char *pz = (char *)tz->buf;
  size_t elem_size = tx->element_size;
  for(size_t i = 0; i < length; i++){
    char *x_ptr = px + i * elem_size;
    char *y_ptr = py + i * elem_size;
    char *z_ptr = pz + i * elem_size;
    switch(dtype){
      case BOOL: *(bool *)z_ptr = *(bool *)x_ptr || *(bool *)y_ptr; break;
      case INT8: *(int8 *)z_ptr = *(int8 *)x_ptr + *(int8 *)y_ptr; break;
      case UINT8: *(uint8 *)z_ptr = *(uint8 *)x_ptr + *(uint8 *)y_ptr; break;
      case INT16: *(int16 *)z_ptr = *(int16 *)x_ptr + *(int16 *)y_ptr; break;
      case UINT16: *(uint16 *)z_ptr = *(uint16 *)x_ptr + *(uint16 *)y_ptr; break;
      case INT32: *(int32 *)z_ptr = *(int32 *)x_ptr + *(int32 *)y_ptr; break;
      case UINT32: *(uint32 *)z_ptr = *(uint32 *)x_ptr + *(uint32 *)y_ptr; break;
      case INT64: *(int64 *)z_ptr = *(int64 *)x_ptr + *(int64 *)y_ptr; break;
      case UINT64: *(uint64 *)z_ptr = *(uint64 *)x_ptr + *(uint64 *)y_ptr; break;
      case FP32: *(float32 *)z_ptr = *(float32 *)x_ptr + *(float32 *)y_ptr; break;
      case FP64: *(float64 *)z_ptr = *(float64 *)x_ptr + *(float64 *)y_ptr; break;
      case FP128: *(float128 *)z_ptr = *(float128 *)x_ptr + *(float128 *)y_ptr; break;
      case CMPX64: {
        complex64 *cx = (complex64 *)x_ptr;
        complex64 *cy = (complex64 *)y_ptr;
        complex64 *cz = (complex64 *)z_ptr;
        cz->real = cx->real + cy->real;
        cz->imag = cx->imag + cy->imag;
        break;
      }
      case CMPX128: {
        complex128 *cx = (complex128 *)x_ptr;
        complex128 *cy = (complex128 *)y_ptr;
        complex128 *cz = (complex128 *)z_ptr;
        cz->real = cx->real + cy->real;
        cz->imag = cx->imag + cy->imag;
        break;
      }
      case ERROR: {
        PyErr_SetString(PyExc_RuntimeError, "something is wrong i can feel it");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *__sub_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  char *px = (char *)tx->buf;
  char *py = (char *)ty->buf;
  char *pz = (char *)tz->buf;
  size_t elem_size = tx->element_size;
  for(size_t i = 0; i < length; i++){
    char *x_ptr = px + i * elem_size;
    char *y_ptr = py + i * elem_size;
    char *z_ptr = pz + i * elem_size;
    switch(dtype){
      case BOOL: *(bool *)z_ptr = *(bool *)x_ptr - *(bool *)y_ptr; break;
      case INT8: *(int8 *)z_ptr = *(int8 *)x_ptr - *(int8 *)y_ptr; break;
      case UINT8: *(uint8 *)z_ptr = *(uint8 *)x_ptr - *(uint8 *)y_ptr; break;
      case INT16: *(int16 *)z_ptr = *(int16 *)x_ptr - *(int16 *)y_ptr; break;
      case UINT16: *(uint16 *)z_ptr = *(uint16 *)x_ptr - *(uint16 *)y_ptr; break;
      case INT32: *(int32 *)z_ptr = *(int32 *)x_ptr - *(int32 *)y_ptr; break;
      case UINT32: *(uint32 *)z_ptr = *(uint32 *)x_ptr - *(uint32 *)y_ptr; break;
      case INT64: *(int64 *)z_ptr = *(int64 *)x_ptr - *(int64 *)y_ptr; break;
      case UINT64: *(uint64 *)z_ptr = *(uint64 *)x_ptr - *(uint64 *)y_ptr; break;
      case FP32: *(float32 *)z_ptr = *(float32 *)x_ptr - *(float32 *)y_ptr; break;
      case FP64: *(float64 *)z_ptr = *(float64 *)x_ptr - *(float64 *)y_ptr; break;
      case FP128: *(float128 *)z_ptr = *(float128 *)x_ptr - *(float128 *)y_ptr; break;
      case CMPX64: {
        complex64 *cx = (complex64 *)x_ptr;
        complex64 *cy = (complex64 *)y_ptr;
        complex64 *cz = (complex64 *)z_ptr;
        cz->real = cx->real - cy->real;
        cz->imag = cx->imag - cy->imag;
        break;
      }
      case CMPX128: {
        complex128 *cx = (complex128 *)x_ptr;
        complex128 *cy = (complex128 *)y_ptr;
        complex128 *cz = (complex128 *)z_ptr;
        cz->real = cx->real - cy->real;
        cz->imag = cx->imag - cy->imag;
        break;
      }
      case ERROR: {
        PyErr_SetString(PyExc_RuntimeError, "something is wrong i can feel it");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *__mul_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  char *px = (char *)tx->buf;
  char *py = (char *)ty->buf;
  char *pz = (char *)tz->buf;
  size_t elem_size = tx->element_size;
  for(size_t i = 0; i < length; i++){
    char *x_ptr = px + i * elem_size;
    char *y_ptr = py + i * elem_size;
    char *z_ptr = pz + i * elem_size;
    switch(dtype){
      case BOOL: *(bool *)z_ptr = *(bool *)x_ptr * *(bool *)y_ptr; break;
      case INT8: *(int8 *)z_ptr = *(int8 *)x_ptr * *(int8 *)y_ptr; break;
      case UINT8: *(uint8 *)z_ptr = *(uint8 *)x_ptr * *(uint8 *)y_ptr; break;
      case INT16: *(int16 *)z_ptr = *(int16 *)x_ptr * *(int16 *)y_ptr; break;
      case UINT16: *(uint16 *)z_ptr = *(uint16 *)x_ptr * *(uint16 *)y_ptr; break;
      case INT32: *(int32 *)z_ptr = *(int32 *)x_ptr * *(int32 *)y_ptr; break;
      case UINT32: *(uint32 *)z_ptr = *(uint32 *)x_ptr * *(uint32 *)y_ptr; break;
      case INT64: *(int64 *)z_ptr = *(int64 *)x_ptr * *(int64 *)y_ptr; break;
      case UINT64: *(uint64 *)z_ptr = *(uint64 *)x_ptr * *(uint64 *)y_ptr; break;
      case FP32: *(float32 *)z_ptr = *(float32 *)x_ptr * *(float32 *)y_ptr; break;
      case FP64: *(float64 *)z_ptr = *(float64 *)x_ptr * *(float64 *)y_ptr; break;
      case FP128: *(float128 *)z_ptr = *(float128 *)x_ptr * *(float128 *)y_ptr; break;
      case CMPX64: {
        complex64 *cx = (complex64 *)x_ptr;
        complex64 *cy = (complex64 *)y_ptr;
        complex64 *cz = (complex64 *)z_ptr;
        cz->real = cx->real * cy->real;
        cz->imag = cx->imag * cy->imag;
        break;
      }
      case CMPX128: {
        complex128 *cx = (complex128 *)x_ptr;
        complex128 *cy = (complex128 *)y_ptr;
        complex128 *cz = (complex128 *)z_ptr;
        cz->real = cx->real * cy->real;
        cz->imag = cx->imag * cy->imag;
        break;
      }
      case ERROR: {
        PyErr_SetString(PyExc_RuntimeError, "something is wrong i can feel it");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static int shapes_match(const tensor_t *tx, const tensor_t *ty){
  if(tx->ndim != ty->ndim) return 0;
  if(!tx->shape || !ty->shape) return 0;
  for(size_t i = 0; i < tx->ndim; i++){
    if(tx->shape[i] != ty->shape[i]) return 0;
  }
  return 1;
}

// Add
static PyObject *add(PyObject *self, PyObject *args){
  PyObject *x, *y;
  if(!PyArg_ParseTuple(args, "OO", &x, &y)) return NULL;
  if(!PyCapsule_CheckExact(x) || !PyCapsule_CheckExact(y)){
    PyErr_SetString(PyExc_RuntimeError, "operands must be the tensor capsule");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  tensor_t *ty = PyCapsule_GetPointer(y, "tensor_t on CPU");
  if(!tx || !ty){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  if(tx->dtype != ty->dtype){
    PyErr_SetString(PyExc_TypeError, "Both tensor_t(s) must have the same dtype");
    return NULL;
  }
  if(tx->device.type != ty->device.type || tx->device.type != CPU){
    PyErr_SetString(PyExc_RuntimeError, "Both tensor_t(s) must be on same device (CPU)");
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
  }else{
    tz = alloc_result_tensor(tx);
    if(!tz) return NULL;
    return __add_tensor__(tx, ty, tz);
  }
}

// Sub
static PyObject *sub(PyObject *self, PyObject *args){
  PyObject *x, *y;
  if(!PyArg_ParseTuple(args, "OO", &x, &y)) return NULL;
  if(!PyCapsule_CheckExact(x) || !PyCapsule_CheckExact(y)){
    PyErr_SetString(PyExc_RuntimeError, "operands must be the tensor capsule");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  tensor_t *ty = PyCapsule_GetPointer(y, "tensor_t on CPU");
  if(!tx || !ty){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  if(tx->dtype != ty->dtype){
    PyErr_SetString(PyExc_TypeError, "Both tensor_t(s) must have the same dtype");
    return NULL;
  }
  if(tx->device.type != ty->device.type || tx->device.type != CPU){
    PyErr_SetString(PyExc_RuntimeError, "Both tensor_t(s) must be on same device (CPU)");
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
    return __sub_tensor__(tx, ty, tz);
  }else{
    tz = alloc_result_tensor(tx);
    if(!tz) return NULL;
    return __sub_tensor__(tx, ty, tz);
  }
}

// Sub
static PyObject *mul(PyObject *self, PyObject *args){
  PyObject *x, *y;
  if(!PyArg_ParseTuple(args, "OO", &x, &y)) return NULL;
  if(!PyCapsule_CheckExact(x) || !PyCapsule_CheckExact(y)){
    PyErr_SetString(PyExc_RuntimeError, "operands must be the tensor capsule");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  tensor_t *ty = PyCapsule_GetPointer(y, "tensor_t on CPU");
  if(!tx || !ty){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  if(tx->dtype != ty->dtype){
    PyErr_SetString(PyExc_TypeError, "Both tensor_t(s) must have the same dtype");
    return NULL;
  }
  if(tx->device.type != ty->device.type || tx->device.type != CPU){
    PyErr_SetString(PyExc_RuntimeError, "Both tensor_t(s) must be on same device (CPU)");
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
    return __mul_tensor__(tx, ty, tz);
  }else{
    tz = alloc_result_tensor(tx);
    if(!tz) return NULL;
    return __mul_tensor__(tx, ty, tz);
  }
}

static PyMethodDef methods[] = {
  {"add", add, METH_VARARGS, "element-wise addition of two tensors or tensor with scalar"},
  {"sub", sub, METH_VARARGS, "element-wise subtraction of two tensors or tensor with scalar"},
  {"mul", mul, METH_VARARGS, "element-wise multiplication of two tensors or tensor with scalar"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "cpu_ops",
  "CPU tensor operations module",
  -1,
  methods
};

PyMODINIT_FUNC PyInit_cpu_ops(void){
  return PyModule_Create(&module);
}
