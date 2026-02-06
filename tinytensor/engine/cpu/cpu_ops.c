// all the kernels assumes dtype homogenity given by the python

#include <python3.10/Python.h>
#include "../tensor.h"

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
      case BOOL: *(bool *)z_ptr = *(bool *)x_ptr & *(bool *)y_ptr; break;
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

static PyObject *__tdiv_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  char *px = (char *)tx->buf;
  char *py = (char *)ty->buf;
  char *pz = (char *)tz->buf;
  size_t elem_size = tx->element_size;
  size_t tz_elem_size = tz->element_size;
  for(size_t i = 0; i < length; i++){
    char *x_ptr = px + i * elem_size;
    char *y_ptr = py + i * elem_size;
    char *z_ptr = pz + i * tz_elem_size;
    switch(dtype){
      case BOOL: *(float32 *)z_ptr = (float32)(*(bool *)x_ptr) / (float32)(*(bool *)y_ptr); break;
      case INT8: *(float32 *)z_ptr = (float32)(*(int8 *)x_ptr) / (float32)(*(int8 *)y_ptr); break;
      case UINT8: *(float32 *)z_ptr = (float32)(*(uint8 *)x_ptr) / (float32)(*(uint8 *)y_ptr); break;
      case INT16: *(float32 *)z_ptr = (float32)(*(int16 *)x_ptr) / (float32)(*(int16 *)y_ptr); break;
      case UINT16: *(float32 *)z_ptr = (float32)(*(uint16 *)x_ptr) / (float32)(*(uint16 *)y_ptr); break;
      case INT32: *(float32 *)z_ptr = (float32)(*(int32 *)x_ptr) / (float32)(*(int32 *)y_ptr); break;
      case UINT32: *(float32 *)z_ptr = (float32)(*(uint32 *)x_ptr) / (float32)(*(uint32 *)y_ptr); break;
      case INT64: *(float32 *)z_ptr = (float32)(*(int64 *)x_ptr) / (float32)(*(int64 *)y_ptr); break;
      case UINT64: *(float32 *)z_ptr = (float32)(*(uint64 *)x_ptr) / (float32)(*(uint64 *)y_ptr); break;
      case FP32: *(float32 *)z_ptr = *(float32 *)x_ptr / *(float32 *)y_ptr; break;
      case FP64: *(float64 *)z_ptr = *(float64 *)x_ptr / *(float64 *)y_ptr; break;
      case ERROR: {
        PyErr_SetString(PyExc_RuntimeError, "something is wrong i can feel it");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *__tdiv_cmpx_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  char *px = (char *)tx->buf;
  char *py = (char *)ty->buf;
  char *pz = (char *)tz->buf;
  size_t elem_size = tx->element_size;
  size_t tz_elem_size = tz->element_size;
  for(size_t i = 0; i < length; i++){
    char *x_ptr = px + i * elem_size;
    char *y_ptr = py + i * elem_size;
    char *z_ptr = pz + i * tz_elem_size;
    switch(dtype){
      case CMPX64: {
        complex64 x = *(complex64 *)x_ptr;
        complex64 y = *(complex64 *)y_ptr;
        float a = x.real;
        float b = x.imag;
        float c = y.real;
        float d = y.imag;
        float denom = c*c + d*d;
        if(denom == 0.0f){
          PyErr_SetString(PyExc_ZeroDivisionError, "complex64 division by zero");
          return NULL;
        }
        complex64 out;
        out.real = (a*c + b*d) / denom;
        out.imag = (b*c - a*d) / denom;
        *(complex64 *)z_ptr = out;
        break;
      }
      case CMPX128: {
        complex128 x = *(complex128 *)x_ptr;
        complex128 y = *(complex128 *)y_ptr;
        double a = x.real;
        double b = x.imag;
        double c = y.real;
        double d = y.imag;
        double denom = c*c + d*d;
        if(denom == 0.0f){
          PyErr_SetString(PyExc_ZeroDivisionError, "complex128 division by zero");
          return NULL;
        }
        complex128 out;
        out.real = (a*c + b*d) / denom;
        out.imag = (b*c - a*d) / denom;
        *(complex128 *)z_ptr = out;
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

static PyObject *__fdiv_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  char *px = (char *)tx->buf;
  char *py = (char *)ty->buf;
  char *pz = (char *)tz->buf;
  size_t elem_size = tx->element_size;
  size_t tz_elem_size = tz->element_size;
  for(size_t i = 0; i < length; i++){
    char *x_ptr = px + i * elem_size;
    char *y_ptr = py + i * elem_size;
    char *z_ptr = pz + i * tz_elem_size;
    switch(dtype){
      case BOOL: *(int64 *)z_ptr = (int64)(*(bool *)x_ptr) / (float32)(*(bool *)y_ptr); break;
      case INT8: *(int64 *)z_ptr = (int64)(*(int8 *)x_ptr) / (float32)(*(int8 *)y_ptr); break;
      case UINT8: *(int64 *)z_ptr = (int64)(*(uint8 *)x_ptr) / (float32)(*(uint8 *)y_ptr); break;
      case INT16: *(int64 *)z_ptr = (int64)(*(int16 *)x_ptr) / (float32)(*(int16 *)y_ptr); break;
      case UINT16: *(int64 *)z_ptr = (int64)(*(uint16 *)x_ptr) / (float32)(*(uint16 *)y_ptr); break;
      case INT32: *(int64 *)z_ptr = (int64)(*(int32 *)x_ptr) / (float32)(*(int32 *)y_ptr); break;
      case UINT32: *(int64 *)z_ptr = (int64)(*(uint32 *)x_ptr) / (float32)(*(uint32 *)y_ptr); break;
      case INT64: *(int64 *)z_ptr = (int64)(*(int64 *)x_ptr) / (float32)(*(int64 *)y_ptr); break;
      case UINT64: *(int64 *)z_ptr = (int64)(*(uint64 *)x_ptr) / (float32)(*(uint64 *)y_ptr); break;
      case FP32: *(int64 *)z_ptr = (int64)(*(float32 *)x_ptr / *(float32 *)y_ptr); break;
      case FP64: *(int64 *)z_ptr = (int64)(*(float64 *)x_ptr / *(float64 *)y_ptr); break;
      case ERROR: {
        PyErr_SetString(PyExc_RuntimeError, "something is wrong i can feel it");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *__pow_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  char *px = (char *)tx->buf;
  char *py = (char *)ty->buf;
  char *pz = (char *)tz->buf;
  size_t elem_size = tx->element_size;
  size_t tz_elem_size = tz->element_size;
  for(size_t i = 0; i < length; i++){
    char *x_ptr = px + i * elem_size;
    char *y_ptr = py + i * elem_size;
    char *z_ptr = pz + i * tz_elem_size;
    switch(dtype){
      case BOOL: *(bool *)z_ptr = (uint8)pow(*(bool *)x_ptr, *(bool *)y_ptr); break;
      case INT8: *(int8 *)z_ptr = (int8)pow(*(int8 *)x_ptr, *(int8 *)y_ptr); break;
      case UINT8: *(uint8 *)z_ptr = (uint8)pow(*(uint8 *)x_ptr, *(uint8 *)y_ptr); break;
      case INT16: *(int16 *)z_ptr = (int16)pow(*(int16 *)x_ptr, *(int16 *)y_ptr); break;
      case UINT16: *(uint16 *)z_ptr = (uint16)pow(*(uint16 *)x_ptr, *(uint16 *)y_ptr); break;
      case INT32: *(int32 *)z_ptr = (int32)pow(*(int32 *)x_ptr, *(int32 *)y_ptr); break;
      case UINT32: *(uint32 *)z_ptr = (uint32)pow(*(uint32 *)x_ptr, *(uint32 *)y_ptr); break;
      case INT64: *(int64 *)z_ptr = (int64)pow(*(int64 *)x_ptr, *(int64 *)y_ptr); break;
      case UINT64: *(uint64 *)z_ptr = (uint64)pow(*(uint64 *)x_ptr, *(uint64 *)y_ptr); break;
      case FP32: *(float32 *)z_ptr = (float32)powf(*(float32 *)x_ptr, *(float32 *)y_ptr); break;
      case FP64: *(float64 *)z_ptr = (float64)pow(*(float64 *)x_ptr, *(float64 *)y_ptr); break;
      case CMPX64: {
        complex64 x = *(complex64 *)x_ptr;
        complex64 y = *(complex64 *)y_ptr;
        float r = sqrt(x.real * x.real + x.imag * x.imag);
        float theta = atan(x.imag/x.real);
        float A = y.real * log(r) - y.imag * theta;
        float B = y.imag * log(r) + y.real * theta;
        float e = exp(A);
        complex64 z = {cos(B), sin(B)};
        (*(complex64 *)z_ptr).real = e * z.real;
        (*(complex64 *)z_ptr).imag = e * z.imag;
        break;
      }
      case CMPX128: {
        complex128 x = *(complex128 *)x_ptr;
        complex128 y = *(complex128 *)y_ptr;
        double r = sqrt(x.real * x.real + x.imag * x.imag);
        double theta = atan(x.imag/x.real);
        double A = y.real * log(r) - y.imag * theta;
        double B = y.imag * log(r) + y.real * theta;
        double e = exp(A);
        complex128 z = {cos(B), sin(B)};
        (*(complex128 *)z_ptr).real = e * z.real;
        (*(complex128 *)z_ptr).imag = e * z.imag;
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

static PyObject *__eq_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  char *px = (char *)tx->buf;
  char *py = (char *)ty->buf;
  char *pz = (char *)tz->buf;
  size_t elem_size = tx->element_size;
  size_t tz_elem_size = tz->element_size;
  for(size_t i = 0; i < length; i++){
    char *x_ptr = px + i * elem_size;
    char *y_ptr = py + i * elem_size;
    char *z_ptr = pz + i * tz_elem_size;
    switch(dtype){
      case BOOL: *(bool *)z_ptr = (bool)(*(bool *)x_ptr == *(bool *)y_ptr); break;
      case INT8: *(bool *)z_ptr = (bool)(*(int8 *)x_ptr == *(int8 *)y_ptr); break;
      case UINT8: *(bool *)z_ptr = (bool)(*(uint8 *)x_ptr == *(uint8 *)y_ptr); break;
      case INT16: *(bool *)z_ptr = (bool)(*(int16 *)x_ptr == *(int16 *)y_ptr); break;
      case UINT16: *(bool *)z_ptr = (bool)(*(uint16 *)x_ptr == *(uint16 *)y_ptr); break;
      case INT32: *(bool *)z_ptr = (bool)(*(int32 *)x_ptr == *(int32 *)y_ptr); break;
      case UINT32: *(bool *)z_ptr = (bool)(*(uint32 *)x_ptr == *(uint32 *)y_ptr); break;
      case INT64: *(bool *)z_ptr = (bool)(*(int64 *)x_ptr == *(int64 *)y_ptr); break;
      case UINT64: *(bool *)z_ptr = (bool)(*(uint64 *)x_ptr == *(uint64 *)y_ptr); break;
      case FP32: *(bool *)z_ptr = (bool)(*(float32 *)x_ptr == *(float32 *)y_ptr); break;
      case FP64: *(bool *)z_ptr = (bool)(*(float64 *)x_ptr == *(float64 *)y_ptr); break;
      case CMPX64: {
        complex64 x = (*(complex64 *)x_ptr);
        complex64 y = (*(complex64 *)y_ptr);
        *(bool *)z_ptr = x.real == y.real && x.imag == y.imag;
        break;
      }
      case CMPX128: {
        complex128 x = (*(complex128 *)x_ptr);
        complex128 y = (*(complex128 *)y_ptr);
        *(bool *)z_ptr = x.real == y.real && x.imag == y.imag;
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

static PyObject *__ne_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  char *px = (char *)tx->buf;
  char *py = (char *)ty->buf;
  char *pz = (char *)tz->buf;
  size_t elem_size = tx->element_size;
  size_t tz_elem_size = tz->element_size;
  for(size_t i = 0; i < length; i++){
    char *x_ptr = px + i * elem_size;
    char *y_ptr = py + i * elem_size;
    char *z_ptr = pz + i * tz_elem_size;
    switch(dtype){
      case BOOL: *(bool *)z_ptr = (bool)(*(bool *)x_ptr != *(bool *)y_ptr); break;
      case INT8: *(bool *)z_ptr = (bool)(*(int8 *)x_ptr != *(int8 *)y_ptr); break;
      case UINT8: *(bool *)z_ptr = (bool)(*(uint8 *)x_ptr != *(uint8 *)y_ptr); break;
      case INT16: *(bool *)z_ptr = (bool)(*(int16 *)x_ptr != *(int16 *)y_ptr); break;
      case UINT16: *(bool *)z_ptr = (bool)(*(uint16 *)x_ptr != *(uint16 *)y_ptr); break;
      case INT32: *(bool *)z_ptr = (bool)(*(int32 *)x_ptr != *(int32 *)y_ptr); break;
      case UINT32: *(bool *)z_ptr = (bool)(*(uint32 *)x_ptr != *(uint32 *)y_ptr); break;
      case INT64: *(bool *)z_ptr = (bool)(*(int64 *)x_ptr != *(int64 *)y_ptr); break;
      case UINT64: *(bool *)z_ptr = (bool)(*(uint64 *)x_ptr != *(uint64 *)y_ptr); break;
      case FP32: *(bool *)z_ptr = (bool)(*(float32 *)x_ptr != *(float32 *)y_ptr); break;
      case FP64: *(bool *)z_ptr = (bool)(*(float64 *)x_ptr != *(float64 *)y_ptr); break;
      case CMPX64: {
        complex64 x = (*(complex64 *)x_ptr);
        complex64 y = (*(complex64 *)y_ptr);
        *(bool *)z_ptr = x.real != y.real || (x.imag != y.imag);
        break;
      }
      case CMPX128:{
        complex128 x = (*(complex128 *)x_ptr);
        complex128 y = (*(complex128 *)y_ptr);
        *(bool *)z_ptr = x.real != y.real || (x.imag != y.imag);
        break;
      }
      case ERROR: {
        PyErr_SetString(PyExc_RuntimeError, "non-equivalance operation not supported for complex dtypes");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *__gt_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  char *px = (char *)tx->buf;
  char *py = (char *)ty->buf;
  char *pz = (char *)tz->buf;
  size_t elem_size = tx->element_size;
  size_t tz_elem_size = tz->element_size;
  for(size_t i = 0; i < length; i++){
    char *x_ptr = px + i * elem_size;
    char *y_ptr = py + i * elem_size;
    char *z_ptr = pz + i * tz_elem_size;
    switch(dtype){
      case BOOL: *(bool *)z_ptr = (bool)(*(bool *)x_ptr > *(bool *)y_ptr); break;
      case INT8: *(bool *)z_ptr = (bool)(*(int8 *)x_ptr > *(int8 *)y_ptr); break;
      case UINT8: *(bool *)z_ptr = (bool)(*(uint8 *)x_ptr > *(uint8 *)y_ptr); break;
      case INT16: *(bool *)z_ptr = (bool)(*(int16 *)x_ptr > *(int16 *)y_ptr); break;
      case UINT16: *(bool *)z_ptr = (bool)(*(uint16 *)x_ptr > *(uint16 *)y_ptr); break;
      case INT32: *(bool *)z_ptr = (bool)(*(int32 *)x_ptr > *(int32 *)y_ptr); break;
      case UINT32: *(bool *)z_ptr = (bool)(*(uint32 *)x_ptr > *(uint32 *)y_ptr); break;
      case INT64: *(bool *)z_ptr = (bool)(*(int64 *)x_ptr > *(int64 *)y_ptr); break;
      case UINT64: *(bool *)z_ptr = (bool)(*(uint64 *)x_ptr > *(uint64 *)y_ptr); break;
      case FP32: *(bool *)z_ptr = (bool)(*(float32 *)x_ptr > *(float32 *)y_ptr); break;
      case FP64: *(bool *)z_ptr = (bool)(*(float64 *)x_ptr > *(float64 *)y_ptr); break;
      case CMPX64:
      case CMPX128:
      case ERROR: {
        PyErr_SetString(PyExc_RuntimeError, "non-equivalance operation not supported for complex dtypes");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *__ge_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  char *px = (char *)tx->buf;
  char *py = (char *)ty->buf;
  char *pz = (char *)tz->buf;
  size_t elem_size = tx->element_size;
  size_t tz_elem_size = tz->element_size;
  for(size_t i = 0; i < length; i++){
    char *x_ptr = px + i * elem_size;
    char *y_ptr = py + i * elem_size;
    char *z_ptr = pz + i * tz_elem_size;
    switch(dtype){
      case BOOL: *(bool *)z_ptr = (bool)(*(bool *)x_ptr >= *(bool *)y_ptr); break;
      case INT8: *(bool *)z_ptr = (bool)(*(int8 *)x_ptr >= *(int8 *)y_ptr); break;
      case UINT8: *(bool *)z_ptr = (bool)(*(uint8 *)x_ptr >= *(uint8 *)y_ptr); break;
      case INT16: *(bool *)z_ptr = (bool)(*(int16 *)x_ptr >= *(int16 *)y_ptr); break;
      case UINT16: *(bool *)z_ptr = (bool)(*(uint16 *)x_ptr >= *(uint16 *)y_ptr); break;
      case INT32: *(bool *)z_ptr = (bool)(*(int32 *)x_ptr >= *(int32 *)y_ptr); break;
      case UINT32: *(bool *)z_ptr = (bool)(*(uint32 *)x_ptr >= *(uint32 *)y_ptr); break;
      case INT64: *(bool *)z_ptr = (bool)(*(int64 *)x_ptr >= *(int64 *)y_ptr); break;
      case UINT64: *(bool *)z_ptr = (bool)(*(uint64 *)x_ptr >= *(uint64 *)y_ptr); break;
      case FP32: *(bool *)z_ptr = (bool)(*(float32 *)x_ptr >= *(float32 *)y_ptr); break;
      case FP64: *(bool *)z_ptr = (bool)(*(float64 *)x_ptr >= *(float64 *)y_ptr); break;
      case CMPX64:
      case CMPX128:
      case ERROR: {
        PyErr_SetString(PyExc_RuntimeError, "non-equivalance operation not supported for complex dtypes");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *__lt_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  char *px = (char *)tx->buf;
  char *py = (char *)ty->buf;
  char *pz = (char *)tz->buf;
  size_t elem_size = tx->element_size;
  size_t tz_elem_size = tz->element_size;
  for(size_t i = 0; i < length; i++){
    char *x_ptr = px + i * elem_size;
    char *y_ptr = py + i * elem_size;
    char *z_ptr = pz + i * tz_elem_size;
    switch(dtype){
      case BOOL: *(bool *)z_ptr = (bool)(*(bool *)x_ptr < *(bool *)y_ptr); break;
      case INT8: *(bool *)z_ptr = (bool)(*(int8 *)x_ptr < *(int8 *)y_ptr); break;
      case UINT8: *(bool *)z_ptr = (bool)(*(uint8 *)x_ptr < *(uint8 *)y_ptr); break;
      case INT16: *(bool *)z_ptr = (bool)(*(int16 *)x_ptr < *(int16 *)y_ptr); break;
      case UINT16: *(bool *)z_ptr = (bool)(*(uint16 *)x_ptr < *(uint16 *)y_ptr); break;
      case INT32: *(bool *)z_ptr = (bool)(*(int32 *)x_ptr < *(int32 *)y_ptr); break;
      case UINT32: *(bool *)z_ptr = (bool)(*(uint32 *)x_ptr < *(uint32 *)y_ptr); break;
      case INT64: *(bool *)z_ptr = (bool)(*(int64 *)x_ptr < *(int64 *)y_ptr); break;
      case UINT64: *(bool *)z_ptr = (bool)(*(uint64 *)x_ptr < *(uint64 *)y_ptr); break;
      case FP32: *(bool *)z_ptr = (bool)(*(float32 *)x_ptr < *(float32 *)y_ptr); break;
      case FP64: *(bool *)z_ptr = (bool)(*(float64 *)x_ptr < *(float64 *)y_ptr); break;
      case CMPX64:
      case CMPX128:
      case ERROR: {
        PyErr_SetString(PyExc_RuntimeError, "non-equivalance operation not supported for complex dtypes");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *__le_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  char *px = (char *)tx->buf;
  char *py = (char *)ty->buf;
  char *pz = (char *)tz->buf;
  size_t elem_size = tx->element_size;
  size_t tz_elem_size = tz->element_size;
  for(size_t i = 0; i < length; i++){
    char *x_ptr = px + i * elem_size;
    char *y_ptr = py + i * elem_size;
    char *z_ptr = pz + i * tz_elem_size;
    switch(dtype){
      case BOOL: *(bool *)z_ptr = (bool)(*(bool *)x_ptr <= *(bool *)y_ptr); break;
      case INT8: *(bool *)z_ptr = (bool)(*(int8 *)x_ptr <= *(int8 *)y_ptr); break;
      case UINT8: *(bool *)z_ptr = (bool)(*(uint8 *)x_ptr <= *(uint8 *)y_ptr); break;
      case INT16: *(bool *)z_ptr = (bool)(*(int16 *)x_ptr <= *(int16 *)y_ptr); break;
      case UINT16: *(bool *)z_ptr = (bool)(*(uint16 *)x_ptr <= *(uint16 *)y_ptr); break;
      case INT32: *(bool *)z_ptr = (bool)(*(int32 *)x_ptr <= *(int32 *)y_ptr); break;
      case UINT32: *(bool *)z_ptr = (bool)(*(uint32 *)x_ptr <= *(uint32 *)y_ptr); break;
      case INT64: *(bool *)z_ptr = (bool)(*(int64 *)x_ptr <= *(int64 *)y_ptr); break;
      case UINT64: *(bool *)z_ptr = (bool)(*(uint64 *)x_ptr <= *(uint64 *)y_ptr); break;
      case FP32: *(bool *)z_ptr = (bool)(*(float32 *)x_ptr <= *(float32 *)y_ptr); break;
      case FP64: *(bool *)z_ptr = (bool)(*(float64 *)x_ptr <= *(float64 *)y_ptr); break;
      case CMPX64:
      case CMPX128:
      case ERROR: {
        destroy(tz);
        PyErr_SetString(PyExc_RuntimeError, "non-equivalance operation not supported for complex dtypes");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

// neg
static PyObject *__neg_tensor__(const tensor_t *tx, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  char *px = (char *)tx->buf;
  char *pz = (char *)tz->buf;
  size_t elem_size = tx->element_size;
  for(size_t i = 0; i < length; i++){
    char *x_ptr = px + i * elem_size;
    char *z_ptr = pz + i * elem_size;
    switch(dtype){
      case INT8: *(int8 *)z_ptr = -(*(int8 *)x_ptr); break;
      case UINT8: *(uint8 *)z_ptr = -(*(uint8 *)x_ptr); break;
      case INT16: *(int16 *)z_ptr = -(*(int16 *)x_ptr); break;
      case UINT16: *(uint16 *)z_ptr = -(*(uint16 *)x_ptr); break;
      case INT32: *(int32 *)z_ptr = -(*(int32 *)x_ptr); break;
      case UINT32: *(uint32 *)z_ptr = -(*(uint32 *)x_ptr); break;
      case INT64: *(int64 *)z_ptr = -(*(int64 *)x_ptr); break;
      case UINT64: *(uint64 *)z_ptr = -(*(uint64 *)x_ptr); break;
      case FP32: *(float32 *)z_ptr = -(*(float32 *)x_ptr); break;
      case FP64: *(float64 *)z_ptr = -(*(float64 *)x_ptr); break;
      case CMPX64: {
        complex64 x = (*(complex64 *)x_ptr);
        (*(complex64 *)z_ptr).real = -x.real;
        (*(complex64 *)z_ptr).imag = -x.imag;
        break;
      }
      case CMPX128: {
        complex128 x = (*(complex128 *)x_ptr);
        (*(complex128 *)z_ptr).real = -x.real;
        (*(complex128 *)z_ptr).imag = -x.imag;
        break;
      }
      case BOOL: {
        PyErr_SetString(PyExc_RuntimeError, "Negation operation on a bool tensor is not supported");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

// pos
static PyObject *__pos_tensor__(const tensor_t *tx, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  char *px = (char *)tx->buf;
  char *pz = (char *)tz->buf;
  size_t elem_size = tx->element_size;
  for(size_t i = 0; i < length; i++){
    char *x_ptr = px + i * elem_size;
    char *z_ptr = pz + i * elem_size;
    switch(dtype){
      case INT8: *(int8 *)z_ptr = *(int8 *)x_ptr; break;
      case UINT8: *(uint8 *)z_ptr = *(uint8 *)x_ptr; break;
      case INT16: *(int16 *)z_ptr = *(int16 *)x_ptr; break;
      case UINT16: *(uint16 *)z_ptr = *(uint16 *)x_ptr; break;
      case INT32: *(int32 *)z_ptr = *(int32 *)x_ptr; break;
      case UINT32: *(uint32 *)z_ptr = *(uint32 *)x_ptr; break;
      case INT64: *(int64 *)z_ptr = *(int64 *)x_ptr; break;
      case UINT64: *(uint64 *)z_ptr = *(uint64 *)x_ptr; break;
      case FP32: *(float32 *)z_ptr = *(float32 *)x_ptr; break;
      case FP64: *(float64 *)z_ptr = *(float64 *)x_ptr; break;
      case CMPX64: *(complex64 *)z_ptr = (*(complex64 *)x_ptr); break;
      case CMPX128: *(complex128 *)z_ptr = *(complex128 *)x_ptr; break;
      case BOOL: {
        PyErr_SetString(PyExc_RuntimeError, "Negation operation on a bool tensor is not supported");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

// abs
static PyObject *__abs_tensor__(const tensor_t *tx, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  char *px = (char *)tx->buf;
  char *pz = (char *)tz->buf;
  size_t elem_size = tx->element_size;
  for(size_t i = 0; i < length; i++){
    char *x_ptr = px + i * elem_size;
    char *z_ptr = pz + i * elem_size;
    switch(dtype){
      case INT8: {
        int8 x = *(int8 *)x_ptr;
        *(int8 *)z_ptr = (x < 0) ? -x : x;
        break;
      }
      case UINT8: *(uint8 *)z_ptr = *(uint8 *)x_ptr; break;
      case INT16: {
        int16 x = *(int16 *)x_ptr;
        *(int16 *)z_ptr = (x < 0) ? -x : x;
        break;
      }
      case UINT16: *(uint16 *)z_ptr = *(uint16 *)x_ptr; break;
      case INT32: {
        int32 x = *(int32 *)x_ptr;
        *(int32 *)z_ptr = (x < 0) ? -x : x;
        break;
      }
      case UINT32: *(uint32 *)z_ptr = *(uint32 *)x_ptr; break;
      case INT64: {
        int64 x = *(int64 *)x_ptr;
        *(int64 *)z_ptr = (x < 0) ? -x : x;
        break;
      }
      case UINT64: *(uint64 *)z_ptr = *(uint64 *)x_ptr; break;
      case FP32: {
        float32 x = *(float32 *)x_ptr;
        *(float32 *)z_ptr = (x < 0) ? -x : x;
        break;
      }
      case FP64: {
        float64 x = *(float64 *)x_ptr;
        *(float64 *)z_ptr = (x < 0) ? -x : x;
        break;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

// abs but for complex tensors
static PyObject *__abs_cmpx_tensor__(const tensor_t *tx, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  char *px = (char *)tx->buf;
  char *pz = (char *)tz->buf;
  size_t tx_elem_size = tx->element_size;
  size_t tz_elem_size = tz->element_size;
  for(size_t i = 0; i < length; i++){
    char *x_ptr = px + i * tx_elem_size;
    char *z_ptr = pz + i * tz_elem_size;
    if(dtype == CMPX64){
      complex64 x = *(complex64 *)x_ptr;
      *(float32 *)z_ptr = (float32)sqrtf(x.real * x.real + x.imag * x.imag);
    }else if(dtype == CMPX128){
      complex128 x = *(complex128 *)x_ptr;
      *(float64 *)z_ptr = (float64)sqrtl(x.real * x.real + x.imag * x.imag);
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

// addition
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
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __add_tensor__(tx, ty, tz);
}

// subtraction
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
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __sub_tensor__(tx, ty, tz);
}

// multiplication
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
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __mul_tensor__(tx, ty, tz);
}

// tdiv
static PyObject *tdiv(PyObject *self, PyObject *args){
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
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    tensor_t *tz = malloc(sizeof(tensor_t));
    if(!tz){
      PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
      return NULL;
    }
    tz->dtype = (tx->dtype == CMPX64) ? CMPX64 : CMPX128;
    tz->size = tx->size;
    tz->ndim = tx->ndim;
    tz->element_size = (tz->dtype == CMPX64) ? sizeof(complex64) : sizeof(complex128);
    tz->device = (device_t){CPU, 0};
    tz->stride = NULL;
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
    return __tdiv_cmpx_tensor__(tx, ty, tz);
  }else{
    tensor_t *tz = malloc(sizeof(tensor_t));
    if(!tz){
      PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
      return NULL;
    }
    tz->dtype = (tx->dtype == FP64) ? FP64 : FP32; // assuming fp32 and fp64 will get the work done
    tz->size = tx->size;
    tz->ndim = tx->ndim;
    tz->element_size = (tz->dtype == FP64) ? sizeof(float64) : sizeof(float32);
    tz->device = (device_t){CPU, 0};
    tz->stride = NULL;
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
    return __tdiv_tensor__(tx, ty, tz);
  }
}

// fdiv
static PyObject *fdiv_(PyObject *self, PyObject *args){
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
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_TypeError, "floor division is not supported on complex tensor(s)");
    return NULL;
  }
  tensor_t *tz = malloc(sizeof(tensor_t));
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
    return NULL;
  }
  tz->dtype = INT64; // assuming fp32 and fp64 will get the work done
  tz->size = tx->size;
  tz->ndim = tx->ndim;
  tz->element_size = sizeof(int64);
  tz->device = (device_t){CPU, 0};
  tz->stride = NULL;
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
  return __fdiv_tensor__(tx, ty, tz);
}

// pow
static PyObject *pow_(PyObject *self, PyObject *args){
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
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __pow_tensor__(tx, ty, tz);
}

// eq
static PyObject *eq(PyObject *self, PyObject *args){
  PyObject *x, *y;
  if(!PyArg_ParseTuple(args, "OO", &x, &y)) return NULL;
  if(!PyCapsule_CheckExact(x) || !PyCapsule_CheckExact(y)){
    PyErr_SetString(PyExc_RuntimeError, "operands must be tensor capsules");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  tensor_t *ty = PyCapsule_GetPointer(y, "tensor_t on CPU");
  if(!tx || !ty){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  if(tx->device.type != CPU || ty->device.type != CPU){
    PyErr_SetString(PyExc_RuntimeError, "Only CPU tensors supported");
    return NULL;
  }
  if(tx->dtype != ty->dtype){
    PyErr_SetString(PyExc_TypeError, "dtype mismatch in eq");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_RuntimeError, "Internal error: size mismatch after broadcasting");
    return NULL;
  }
  tensor_t *tz = malloc(sizeof(tensor_t));
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
    return NULL;
  }
  tz->dtype = BOOL;
  tz->size = tx->size;
  tz->ndim = tx->ndim;
  tz->element_size = sizeof(bool);
  tz->device = (device_t){CPU, 0};
  tz->stride = NULL;
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
  return __eq_tensor__(tx, ty, tz);
}

// ne
static PyObject *ne(PyObject *self, PyObject *args){
  PyObject *x, *y;
  if(!PyArg_ParseTuple(args, "OO", &x, &y)) return NULL;
  if(!PyCapsule_CheckExact(x) || !PyCapsule_CheckExact(y)){
    PyErr_SetString(PyExc_RuntimeError, "operands must be tensor capsules");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  tensor_t *ty = PyCapsule_GetPointer(y, "tensor_t on CPU");
  if(!tx || !ty){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  if(tx->device.type != CPU || ty->device.type != CPU){
    PyErr_SetString(PyExc_RuntimeError, "Only CPU tensors supported");
    return NULL;
  }
  if(tx->dtype != ty->dtype){
    PyErr_SetString(PyExc_TypeError, "dtype mismatch in eq");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_RuntimeError, "Internal error: size mismatch after broadcasting");
    return NULL;
  }
  tensor_t *tz = malloc(sizeof(tensor_t));
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
    return NULL;
  }
  tz->dtype = BOOL;
  tz->size = tx->size;
  tz->ndim = tx->ndim;
  tz->element_size = sizeof(bool);
  tz->device = (device_t){CPU, 0};
  tz->stride = NULL;
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
  return __ne_tensor__(tx, ty, tz);
}

// gt
static PyObject *gt(PyObject *self, PyObject *args){
  PyObject *x, *y;
  if(!PyArg_ParseTuple(args, "OO", &x, &y)) return NULL;
  if(!PyCapsule_CheckExact(x) || !PyCapsule_CheckExact(y)){
    PyErr_SetString(PyExc_RuntimeError, "operands must be tensor capsules");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  tensor_t *ty = PyCapsule_GetPointer(y, "tensor_t on CPU");
  if(!tx || !ty){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  if(tx->device.type != CPU || ty->device.type != CPU){
    PyErr_SetString(PyExc_RuntimeError, "Only CPU tensors supported");
    return NULL;
  }
  if(tx->dtype != ty->dtype){
    PyErr_SetString(PyExc_TypeError, "dtype mismatch in eq");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "> operation on complex tensor is not supported");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_RuntimeError, "Internal error: size mismatch after broadcasting");
    return NULL;
  }
  tensor_t *tz = malloc(sizeof(tensor_t));
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
    return NULL;
  }
  tz->dtype = BOOL;
  tz->size = tx->size;
  tz->ndim = tx->ndim;
  tz->element_size = sizeof(bool);
  tz->device = (device_t){CPU, 0};
  tz->stride = NULL;
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
  return __gt_tensor__(tx, ty, tz);
}

// ge
static PyObject *ge(PyObject *self, PyObject *args){
  PyObject *x, *y;
  if(!PyArg_ParseTuple(args, "OO", &x, &y)) return NULL;
  if(!PyCapsule_CheckExact(x) || !PyCapsule_CheckExact(y)){
    PyErr_SetString(PyExc_RuntimeError, "operands must be tensor capsules");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  tensor_t *ty = PyCapsule_GetPointer(y, "tensor_t on CPU");
  if(!tx || !ty){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  if(tx->device.type != CPU || ty->device.type != CPU){
    PyErr_SetString(PyExc_RuntimeError, "Only CPU tensors supported");
    return NULL;
  }
  if(tx->dtype != ty->dtype){
    PyErr_SetString(PyExc_TypeError, "dtype mismatch in eq");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, ">= operation on complex tensor is not supported");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_RuntimeError, "Internal error: size mismatch after broadcasting");
    return NULL;
  }
  tensor_t *tz = malloc(sizeof(tensor_t));
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
    return NULL;
  }
  tz->dtype = BOOL;
  tz->size = tx->size;
  tz->ndim = tx->ndim;
  tz->element_size = sizeof(bool);
  tz->device = (device_t){CPU, 0};
  tz->stride = NULL;
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
  return __ge_tensor__(tx, ty, tz);
}

// lt
static PyObject *lt(PyObject *self, PyObject *args){
  PyObject *x, *y;
  if(!PyArg_ParseTuple(args, "OO", &x, &y)) return NULL;
  if(!PyCapsule_CheckExact(x) || !PyCapsule_CheckExact(y)){
    PyErr_SetString(PyExc_RuntimeError, "operands must be tensor capsules");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  tensor_t *ty = PyCapsule_GetPointer(y, "tensor_t on CPU");
  if(!tx || !ty){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  if(tx->device.type != CPU || ty->device.type != CPU){
    PyErr_SetString(PyExc_RuntimeError, "Only CPU tensors supported");
    return NULL;
  }
  if(tx->dtype != ty->dtype){
    PyErr_SetString(PyExc_TypeError, "dtype mismatch in eq");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "< operation on complex tensor is not supported");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_RuntimeError, "Internal error: size mismatch after broadcasting");
    return NULL;
  }
  tensor_t *tz = malloc(sizeof(tensor_t));
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
    return NULL;
  }
  tz->dtype = BOOL;
  tz->size = tx->size;
  tz->ndim = tx->ndim;
  tz->element_size = sizeof(bool);
  tz->device = (device_t){CPU, 0};
  tz->stride = NULL;
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
  return __lt_tensor__(tx, ty, tz);
}

// le
static PyObject *le(PyObject *self, PyObject *args){
  PyObject *x, *y;
  if(!PyArg_ParseTuple(args, "OO", &x, &y)) return NULL;
  if(!PyCapsule_CheckExact(x) || !PyCapsule_CheckExact(y)){
    PyErr_SetString(PyExc_RuntimeError, "operands must be tensor capsules");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  tensor_t *ty = PyCapsule_GetPointer(y, "tensor_t on CPU");
  if(!tx || !ty){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  if(tx->device.type != CPU || ty->device.type != CPU){
    PyErr_SetString(PyExc_RuntimeError, "Only CPU tensors supported");
    return NULL;
  }
  if(tx->dtype != ty->dtype){
    PyErr_SetString(PyExc_TypeError, "dtype mismatch in eq");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "<= operation on complex tensor is not supported");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_RuntimeError, "Internal error: size mismatch after broadcasting");
    return NULL;
  }
  tensor_t *tz = malloc(sizeof(tensor_t));
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
    return NULL;
  }
  tz->dtype = BOOL;
  tz->size = tx->size;
  tz->ndim = tx->ndim;
  tz->element_size = sizeof(bool);
  tz->device = (device_t){CPU, 0};
  tz->stride = NULL;
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
  return __le_tensor__(tx, ty, tz);
}

// neg
static PyObject *neg(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "operands must be tensor capsules");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  if(tx->dtype == BOOL){
    PyErr_SetString(PyExc_RuntimeError, "Negation operartion on bool tensor is not supported");
    return NULL;
  }
  if(tx->device.type != CPU){
    PyErr_SetString(PyExc_RuntimeError, "Only CPU tensors supported");
    return NULL;
  }
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __neg_tensor__(tx, tz);
}

// pos
static PyObject *pos(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "operands must be tensor capsules");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  if(tx->dtype == BOOL){
    PyErr_SetString(PyExc_RuntimeError, "Negation operartion on bool tensor is not supported");
    return NULL;
  }
  if(tx->device.type != CPU){
    PyErr_SetString(PyExc_RuntimeError, "Only CPU tensors supported");
    return NULL;
  }
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __pos_tensor__(tx, tz);
}

// abs
static PyObject *abs_(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "operands must be tensor capsules");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  if(tx->dtype == BOOL){
    PyErr_SetString(PyExc_RuntimeError, "abs() on bool tensor is not supported");
    return NULL;
  }
  if(tx->device.type != CPU){
    PyErr_SetString(PyExc_RuntimeError, "Only CPU tensors supported");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    tensor_t *tz = malloc(sizeof(tensor_t));
    if(!tz){
      PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
      return NULL;
    }
    tz->dtype = (tx->dtype == CMPX64) ? FP32 : FP64;
    tz->size = tx->size;
    tz->ndim = tx->ndim;
    tz->element_size = getsize(tz->dtype);
    tz->device = tx->device;
    tz->stride = NULL;
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
    return __abs_cmpx_tensor__(tx, tz);
  }else{
    tensor_t *tz = NULL;
    tz = alloc_result_tensor(tx);
    return __abs_tensor__(tx, tz);
  }
}

static PyMethodDef methods[] = {
  {"add", add, METH_VARARGS, "element-wise addition of two tensors or tensor with scalar"},
  {"sub", sub, METH_VARARGS, "element-wise subtraction of two tensors or tensor with scalar"},
  {"mul", mul, METH_VARARGS, "element-wise multiplication of two tensors or tensor with scalar"},
  {"tdiv", tdiv, METH_VARARGS, "element-wise true division of two tensors or tensor with scalar"},
  {"fdiv", fdiv_, METH_VARARGS, "element-wise floor division of two tensors or tensor with scalar"},
  {"pow", pow_, METH_VARARGS, "element-wise powe operation of two tensors or tensor with sclaar"},
  {"eq", eq, METH_VARARGS, "element-wise equivalance of two tensors or tensor with scalar"},
  {"ne", ne, METH_VARARGS, "element-wise non-equivalance of two tensors or tensor with scalar"},
  {"gt", gt, METH_VARARGS, "element-wise greater than op of two tensors or tensor with scalar"},
  {"ge", ge, METH_VARARGS, "element-wise greater than equal op of two tensors or tensor with scalar"},
  {"lt", lt, METH_VARARGS, "element-wise less than op of two tensors or tensor with scalar"},
  {"le", le, METH_VARARGS, "element-wise less than equal op of two tensors or tensor with scalar"},
  {"neg", neg, METH_VARARGS, "negation operation on tensor"},
  {"pos", pos, METH_VARARGS, "non-negation operation on tensor"},
  {"abs", abs_, METH_VARARGS, "absolute value of tensor or tensor with scalar"},
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
