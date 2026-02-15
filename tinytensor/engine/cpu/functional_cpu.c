#include <python3.10/Python.h>
#include "../tensor.h"

#define SQRT_2_DIV_PI 0.79788456080286535587989211986876

static inline double gelu(double x){
  double x3 = x * x * x;
  return 0.5 * x * (1.0 + tanh(SQRT_2_DIV_PI * (x + 0.044715 * x3)));
}
static inline double leaky_relu(double x, double alpha){ return x > 0.0 ? x : alpha * x; }

void capsule_destroyer(PyObject *capsule){
  tensor_t *t = PyCapsule_GetPointer(capsule, "tensor_t on CPU");
  if(t){
    destroy(t);
  }
}

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
  tz->stride = malloc(sizeof(size_t) * tz->ndim);
  if(!tz->stride){ return NULL; }
  for(size_t i = 0; i < tz->ndim; i++){ tz->stride[i] = t->stride[i]; }
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

static void __relu__(const tensor_t *tx, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  switch(dtype){
    case INT8: {
      int8 *x = (int8 *)tx->buf;
      int8 *z = (int8 *)tz->buf;
      for(size_t i = 0; i < length; i++){ z[i] = (x[i] > 0) ? x[i] : 0; }
      break;
    }
    case INT16: {
      int16 *x = (int16 *)tx->buf;
      int16 *z = (int16 *)tz->buf;
      for(size_t i = 0; i < length; i++){ z[i] = (x[i] > 0) ? x[i] : 0; }
      break;
    }
    case INT32: {
      int32 *x = (int32 *)tx->buf;
      int32 *z = (int32 *)tz->buf;
      for(size_t i = 0; i < length; i++){ z[i] = (x[i] > 0) ? x[i] : 0; }
      break;
    }
    case INT64: {
      int64 *x = (int64 *)tx->buf;
      int64 *z = (int64 *)tz->buf;
      for(size_t i = 0; i < length; i++){ z[i] = (x[i] > 0) ? x[i] : 0; }
      break;
    }
    case FP32: {
      float32 *x = (float32 *)tx->buf;
      float32 *z = (float32 *)tz->buf;
      for(size_t i = 0; i < length; i++){ z[i] = (x[i] > 0) ? x[i] : 0; }
      break;
    }
    case FP64: {
      float64 *x = (float64 *)tx->buf;
      float64 *z = (float64 *)tz->buf;
      for(size_t i = 0; i < length; i++){ z[i] = (x[i] > 0) ? x[i] : 0; }
      break;
    }
    case UINT8:
    case UINT16:
    case UINT32:
    case UINT64: { memcpy(tz->buf, tx->buf, tx->size * tx->element_size); break; }
    default: {
      PyErr_SetString(PyExc_RuntimeError, "__relu__: unsupported dtype");
      break;
    }
  }
}

static void __gelu__(const tensor_t *tx, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  switch(dtype){
    case INT8: {
      int8 *x = (int8 *)tx->buf;
      float32 *z = (float32 *)tz->buf;
      for(size_t i = 0; i < length; i++){ z[i] = (float32)gelu((float32)x[i]); }
      break;
    }
    case UINT8: {
      uint8 *x = (uint8 *)tx->buf;
      float32 *z = (float32 *)tz->buf;
      for(size_t i = 0; i < length; i++){ z[i] = (float32)gelu((float32)x[i]); }
      break;
    }
    case INT16: {
      int16 *x = (int16 *)tx->buf;
      float32 *z = (float32 *)tz->buf;
      for(size_t i = 0; i < length; i++){ z[i] = (float32)gelu((float32)x[i]); }
      break;
    }
    case UINT16: {
      uint16 *x = (uint16 *)tx->buf;
      float32 *z = (float32 *)tz->buf;
      for(size_t i = 0; i < length; i++){ z[i] = (float32)gelu((float32)x[i]);}
      break;
    }
    case INT32: {
      int32 *x = (int32 *)tx->buf;
      float32 *z = (float32 *)tz->buf;
      for(size_t i = 0; i < length; i++){ z[i] = (float32)gelu((float32)x[i]);}
      break;
    }
    case UINT32: {
      uint32 *x = (uint32 *)tx->buf;
      float32 *z = (float32 *)tz->buf;
      for(size_t i = 0; i < length; i++){ z[i] = (float32)gelu((float32)x[i]); }
      break;
    }
    case INT64: {
      int64 *x = (int64 *)tx->buf;
      float32 *z = (float32 *)tz->buf;
      for(size_t i = 0; i < length; i++){ z[i] = (float32)gelu((float64)x[i]); }
      break;
    }
    case UINT64: {
      uint64 *x = (uint64 *)tx->buf;
      float32 *z = (float32 *)tz->buf;
      for(size_t i = 0; i < length; i++){ z[i] = (float32)gelu((float64)x[i]); }
      break;
    }
    case FP32: {
      float32 *x = (float32 *)tx->buf;
      float32 *z = (float32 *)tz->buf;
      for(size_t i = 0; i < length; i++){ z[i] = (float32)gelu((float64)x[i]); }
      break;
    }
    case FP64: {
      float64 *x = (float64 *)tx->buf;
      float64 *z = (float64 *)tz->buf;
      for(size_t i = 0; i < length; i++){ z[i] = gelu(x[i]); }
      break;
    }
    default: {
      PyErr_SetString(PyExc_RuntimeError, "__gelu__: unsupported dtype");
      break;
    }
  }
}

static void __leaky_relu__(const tensor_t *tx, tensor_t *tz, double alpha){
  size_t length = tx->size;
  switch(tx->dtype){
    case INT8: {
      int8 *x = (int8 *)tx->buf;
      float32 *z = (float32 *)tz->buf;
      for(size_t i = 0; i < length; i++)
        z[i] = (float32)leaky_relu((float64)x[i], alpha);
      break;
    }
    case INT16: {
      int16 *x = (int16 *)tx->buf;
      float32 *z = (float32 *)tz->buf;
      for(size_t i = 0; i < length; i++)
        z[i] = (float32)leaky_relu((float64)x[i], alpha);
      break;
    }
    case INT32: {
      int32 *x = (int32 *)tx->buf;
      float32 *z = (float32 *)tz->buf;
      for(size_t i = 0; i < length; i++)
        z[i] = (float32)leaky_relu((float64)x[i], alpha);
      break;
    }
    case INT64: {
      int64 *x = (int64 *)tx->buf;
      float32 *z = (float32 *)tz->buf;
      for(size_t i = 0; i < length; i++)
        z[i] = (float32)leaky_relu((float64)x[i], alpha);
      break;
    }
    case UINT8: {
      uint8 *x = (uint8 *)tx->buf;
      float32 *z = (float32 *)tz->buf;
      for(size_t i = 0; i < length; i++)
        z[i] = (float32)leaky_relu((float64)x[i], alpha);
      break;
    }
    case UINT16: {
      uint16 *x = (uint16 *)tx->buf;
      float32 *z = (float32 *)tz->buf;
      for(size_t i = 0; i < length; i++)
        z[i] = (float32)leaky_relu((float64)x[i], alpha);
      break;
    }
    case UINT32: {
      uint32 *x = (uint32 *)tx->buf;
      float32 *z = (float32 *)tz->buf;
      for(size_t i = 0; i < length; i++)
        z[i] = (float32)leaky_relu((float64)x[i], alpha);
      break;
    }
    case UINT64: {
      uint64 *x = (uint64 *)tx->buf;
      float64 *z = (float64 *)tz->buf;
      for(size_t i = 0; i < length; i++)
        z[i] = leaky_relu((float64)x[i], alpha);
      break;
    }
    case FP32: {
      float32 *x = (float32 *)tx->buf;
      float32 *z = (float32 *)tz->buf;
      for(size_t i = 0; i < length; i++)
        z[i] = (float32)leaky_relu((float64)x[i], alpha);
      break;
    }
    case FP64: {
      float64 *x = (float64 *)tx->buf;
      float64 *z = (float64 *)tz->buf;
      for(size_t i = 0; i < length; i++)
        z[i] = leaky_relu(x[i], alpha);
      break;
    }
    default:
      PyErr_SetString(PyExc_RuntimeError, "__leaky_relu__: unsupported dtype");
  }
}

static PyObject *relu_act(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid capsule pointer");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "relu is not supported for complex dtype tensors");
    return NULL;
  }
  tensor_t *tz = alloc_result_tensor(tx);
  __relu__(tx, tz);
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *gelu_act(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid capsule pointer");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "relu is not supported for complex dtype tensors");
    return NULL;
  }
  tensor_t *tz = malloc(sizeof(tensor_t));
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
    return NULL;
  }
  tz->dtype = (tx->dtype == FP64) ? FP64 : FP32;
  tz->size = tx->size;
  tz->ndim = tx->ndim;
  tz->element_size = (tz->dtype == FP64) ? sizeof(float64) : sizeof(float32);
  tz->device = tx->device;
  tz->stride = malloc(sizeof(size_t) * tz->ndim);
  if(!tz->stride){ return NULL; }
  for(size_t i = 0; i < tz->ndim; i++){ tz->stride[i] = tx->stride[i]; }
  if(tx->ndim > 0 && tx->shape){
    tz->shape = malloc(sizeof(size_t) * tz->ndim);
    if(!tz->shape){
      free(tz);
      PyErr_SetString(PyExc_RuntimeError, "tensor_t.shape allocation failed!");
      return NULL;
    }
    for(size_t i = 0; i < tz->ndim; i++){
      tz->shape[i] = tx->shape[i];
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
  __gelu__(tx, tz);
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *leaky_relu_act(PyObject *self, PyObject *args){
  PyObject *x;
  double alpha = 0.01;
  if(!PyArg_ParseTuple(args, "O|d", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid capsule pointer");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "relu is not supported for complex dtype tensors");
    return NULL;
  }
  tensor_t *tz = malloc(sizeof(tensor_t));
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
    return NULL;
  }
  tz->dtype = (tx->dtype == FP64) ? FP64 : FP32;
  tz->size = tx->size;
  tz->ndim = tx->ndim;
  tz->element_size = (tz->dtype == FP64) ? sizeof(float64) : sizeof(float32);
  tz->device = tx->device;
  tz->stride = malloc(sizeof(size_t) * tz->ndim);
  if(!tz->stride){ return NULL; }
  for(size_t i = 0; i < tz->ndim; i++){ tz->stride[i] = tx->stride[i]; }
  if(tx->ndim > 0 && tx->shape){
    tz->shape = malloc(sizeof(size_t) * tz->ndim);
    if(!tz->shape){
      free(tz);
      PyErr_SetString(PyExc_RuntimeError, "tensor_t.shape allocation failed!");
      return NULL;
    }
    for(size_t i = 0; i < tz->ndim; i++){
      tz->shape[i] = tx->shape[i];
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
  __leaky_relu__(tx, tz, alpha);
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyMethodDef methods[] = {
  {"relu", relu_act, METH_VARARGS, "element-wise 'relu' operation on tensor"},
  {"gelu", gelu_act, METH_VARARGS, "element-wise 'gelu' operation on tensor"},
  {"leaky_relu", leaky_relu_act, METH_VARARGS, "element-wise 'leaky_relu' operation on tensor"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "functional_cpu",
  "CPU tensor functional operations module",
  -1,
  methods
};

PyMODINIT_FUNC PyInit_functional_cpu(void){
  return PyModule_Create(&module);
}
