#include <python3.10/Python.h>
#include <python3.10/methodobject.h>
#include "../tensor.h"

void capsule_destroyer(PyObject *capsule){
  tensor_t *t = PyCapsule_GetPointer(capsule, "tensor_t on CPU");
  if(t){
    destroy(t);
  }
}

tensor_t *tensor_empty_scalar_like(tensor_t *tx){
  tensor_t *tz = malloc(sizeof(tensor_t));
  if(!tz) return NULL;
  tz->dtype = tx->dtype;
  tz->device = tx->device;
  tz->element_size = tx->element_size;
  tz->ndim = 0;
  tz->size = 1;
  tz->shape = NULL;
  tz->stride = NULL;
  tz->storage = malloc(sizeof(storage_t));
  if(!tz->storage){
    free(tz);
    return NULL;
  }
  tz->storage->bytes = tz->element_size;
  tz->storage->device = tz->device;
  tz->storage->refcount = 1;
  tz->storage->ptr = calloc(1, tz->storage->bytes);
  if(!tz->storage->ptr){
    free(tz->storage);
    free(tz);
    return NULL;
  }
  tz->buf = tz->storage->ptr;
  return tz;
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

static PyObject *__mod_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
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
      case BOOL: *(bool *)z_ptr = (uint8)fmodf(*(bool *)x_ptr, (*(bool *)y_ptr)); break;
      case INT8: *(int8 *)z_ptr = (int8)fmodf(*(int8 *)x_ptr, *(int8 *)y_ptr); break;
      case UINT8: *(uint8 *)z_ptr = (uint8)fmodf(*(uint8 *)x_ptr, *(uint8 *)y_ptr); break;
      case INT16: *(int16 *)z_ptr = (int16)fmodf(*(int16 *)x_ptr, *(int16 *)y_ptr); break;
      case UINT16: *(uint16 *)z_ptr = (uint16)fmodf(*(uint16 *)x_ptr, *(uint16 *)y_ptr); break;
      case INT32: *(int32 *)z_ptr = (int32)fmodf(*(int32 *)x_ptr, *(int32 *)y_ptr); break;
      case UINT32: *(uint32 *)z_ptr = (uint32)fmodf(*(uint32 *)x_ptr, *(uint32 *)y_ptr); break;
      case INT64: *(int64 *)z_ptr = (int64)fmodf(*(int64 *)x_ptr, *(int64 *)y_ptr); break;
      case UINT64: *(uint64 *)z_ptr = (uint64)fmodf(*(uint64 *)x_ptr, *(uint64 *)y_ptr); break;
      case FP32: *(float32 *)z_ptr = (float32)fmodf(*(float32 *)x_ptr, *(float32 *)y_ptr); break;
      case FP64: *(float64 *)z_ptr = (float64)fmodl(*(float64 *)x_ptr, *(float64 *)y_ptr); break;
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

// left shift
static PyObject *__lshift_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
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
      case INT8: *(int8 *)z_ptr = *(int8 *)x_ptr << *(int8 *)y_ptr; break;
      case UINT8: *(uint8 *)z_ptr = *(uint8 *)x_ptr << *(uint8 *)y_ptr; break;
      case INT16: *(int16 *)z_ptr = *(int16 *)x_ptr << *(int16 *)y_ptr; break;
      case UINT16: *(uint16 *)z_ptr = *(uint16 *)x_ptr << *(uint16 *)y_ptr; break;
      case INT32: *(int32 *)z_ptr = *(int32 *)x_ptr << *(int32 *)y_ptr; break;
      case UINT32: *(uint32 *)z_ptr = *(uint32 *)x_ptr << *(uint32 *)y_ptr; break;
      case INT64: *(int64 *)z_ptr = *(int64 *)x_ptr << *(int64 *)y_ptr; break;
      case UINT64: *(uint64 *)z_ptr = *(uint64 *)x_ptr << *(uint64 *)y_ptr; break;
      case ERROR: {
        PyErr_SetString(PyExc_RuntimeError, "something is wrong i can feel it");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

// right shift
static PyObject *__rshift_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
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
      case INT8: *(int8 *)z_ptr = *(int8 *)x_ptr >> *(int8 *)y_ptr; break;
      case UINT8: *(uint8 *)z_ptr = *(uint8 *)x_ptr >> *(uint8 *)y_ptr; break;
      case INT16: *(int16 *)z_ptr = *(int16 *)x_ptr >> *(int16 *)y_ptr; break;
      case UINT16: *(uint16 *)z_ptr = *(uint16 *)x_ptr >> *(uint16 *)y_ptr; break;
      case INT32: *(int32 *)z_ptr = *(int32 *)x_ptr >> *(int32 *)y_ptr; break;
      case UINT32: *(uint32 *)z_ptr = *(uint32 *)x_ptr >> *(uint32 *)y_ptr; break;
      case INT64: *(int64 *)z_ptr = *(int64 *)x_ptr >> *(int64 *)y_ptr; break;
      case UINT64: *(uint64 *)z_ptr = *(uint64 *)x_ptr >> *(uint64 *)y_ptr; break;
      case ERROR: {
        PyErr_SetString(PyExc_RuntimeError, "something is wrong i can feel it");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

// and
static PyObject *__and_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
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
      case INT8: *(int8 *)z_ptr = *(int8 *)x_ptr & *(int8 *)y_ptr; break;
      case UINT8: *(uint8 *)z_ptr = *(uint8 *)x_ptr & *(uint8 *)y_ptr; break;
      case INT16: *(int16 *)z_ptr = *(int16 *)x_ptr & *(int16 *)y_ptr; break;
      case UINT16: *(uint16 *)z_ptr = *(uint16 *)x_ptr & *(uint16 *)y_ptr; break;
      case INT32: *(int32 *)z_ptr = *(int32 *)x_ptr & *(int32 *)y_ptr; break;
      case UINT32: *(uint32 *)z_ptr = *(uint32 *)x_ptr & *(uint32 *)y_ptr; break;
      case INT64: *(int64 *)z_ptr = *(int64 *)x_ptr & *(int64 *)y_ptr; break;
      case UINT64: *(uint64 *)z_ptr = *(uint64 *)x_ptr & *(uint64 *)y_ptr; break;
      case ERROR: {
        PyErr_SetString(PyExc_RuntimeError, "something is wrong i can feel it");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

// and
static PyObject *__nand_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
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
      case INT8: *(int8 *)z_ptr = ~(*(int8 *)x_ptr & *(int8 *)y_ptr); break;
      case UINT8: *(uint8 *)z_ptr = ~(*(uint8 *)x_ptr & *(uint8 *)y_ptr); break;
      case INT16: *(int16 *)z_ptr = ~(*(int16 *)x_ptr & *(int16 *)y_ptr); break;
      case UINT16: *(uint16 *)z_ptr = ~(*(uint16 *)x_ptr & *(uint16 *)y_ptr); break;
      case INT32: *(int32 *)z_ptr = ~(*(int32 *)x_ptr & *(int32 *)y_ptr); break;
      case UINT32: *(uint32 *)z_ptr = ~(*(uint32 *)x_ptr & *(uint32 *)y_ptr); break;
      case INT64: *(int64 *)z_ptr = ~(*(int64 *)x_ptr & *(int64 *)y_ptr); break;
      case UINT64: *(uint64 *)z_ptr = ~(*(uint64 *)x_ptr & *(uint64 *)y_ptr); break;
      case ERROR: {
        PyErr_SetString(PyExc_RuntimeError, "something is wrong i can feel it");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

// or
static PyObject *__or_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
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
      case INT8: *(int8 *)z_ptr = *(int8 *)x_ptr | *(int8 *)y_ptr; break;
      case UINT8: *(uint8 *)z_ptr = *(uint8 *)x_ptr | *(uint8 *)y_ptr; break;
      case INT16: *(int16 *)z_ptr = *(int16 *)x_ptr | *(int16 *)y_ptr; break;
      case UINT16: *(uint16 *)z_ptr = *(uint16 *)x_ptr | *(uint16 *)y_ptr; break;
      case INT32: *(int32 *)z_ptr = *(int32 *)x_ptr | *(int32 *)y_ptr; break;
      case UINT32: *(uint32 *)z_ptr = *(uint32 *)x_ptr | *(uint32 *)y_ptr; break;
      case INT64: *(int64 *)z_ptr = *(int64 *)x_ptr | *(int64 *)y_ptr; break;
      case UINT64: *(uint64 *)z_ptr = *(uint64 *)x_ptr | *(uint64 *)y_ptr; break;
      case ERROR: {
        PyErr_SetString(PyExc_RuntimeError, "something is wrong i can feel it");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

// nor
static PyObject *__nor_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
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
      case INT8: *(int8 *)z_ptr = ~(*(int8 *)x_ptr | *(int8 *)y_ptr); break;
      case UINT8: *(uint8 *)z_ptr = ~(*(uint8 *)x_ptr | *(uint8 *)y_ptr); break;
      case INT16: *(int16 *)z_ptr = ~(*(int16 *)x_ptr | *(int16 *)y_ptr); break;
      case UINT16: *(uint16 *)z_ptr = ~(*(uint16 *)x_ptr | *(uint16 *)y_ptr); break;
      case INT32: *(int32 *)z_ptr = ~(*(int32 *)x_ptr | *(int32 *)y_ptr); break;
      case UINT32: *(uint32 *)z_ptr = ~(*(uint32 *)x_ptr | *(uint32 *)y_ptr); break;
      case INT64: *(int64 *)z_ptr = ~(*(int64 *)x_ptr | *(int64 *)y_ptr); break;
      case UINT64: *(uint64 *)z_ptr = ~(*(uint64 *)x_ptr | *(uint64 *)y_ptr); break;
      case ERROR: {
        PyErr_SetString(PyExc_RuntimeError, "something is wrong i can feel it");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

// not
static PyObject *__not_tensor__(const tensor_t *tx, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  char *px = (char *)tx->buf;
  char *pz = (char *)tz->buf;
  size_t elem_size = tx->element_size;
  for(size_t i = 0; i < length; i++){
    char *x_ptr = px + i * elem_size;
    char *z_ptr = pz + i * elem_size;
    switch(dtype){
      case INT8: *(int8 *)z_ptr = ~(*(int8 *)x_ptr); break;
      case UINT8: *(uint8 *)z_ptr = ~(*(uint8 *)x_ptr); break;
      case INT16: *(int16 *)z_ptr = ~(*(int16 *)x_ptr); break;
      case UINT16: *(uint16 *)z_ptr = ~(*(uint16 *)x_ptr); break;
      case INT32: *(int32 *)z_ptr = ~(*(int32 *)x_ptr); break;
      case UINT32: *(uint32 *)z_ptr = ~(*(uint32 *)x_ptr); break;
      case INT64: *(int64 *)z_ptr = ~(*(int64 *)x_ptr); break;
      case UINT64: *(uint64 *)z_ptr = ~(*(uint64 *)x_ptr); break;
      case ERROR: {
        PyErr_SetString(PyExc_RuntimeError, "something is wrong i can feel it");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

// xor
static PyObject *__xor_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
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
      case INT8: *(int8 *)z_ptr = *(int8 *)x_ptr ^ *(int8 *)y_ptr; break;
      case UINT8: *(uint8 *)z_ptr = *(uint8 *)x_ptr ^ *(uint8 *)y_ptr; break;
      case INT16: *(int16 *)z_ptr = *(int16 *)x_ptr ^ *(int16 *)y_ptr; break;
      case UINT16: *(uint16 *)z_ptr = *(uint16 *)x_ptr ^ *(uint16 *)y_ptr; break;
      case INT32: *(int32 *)z_ptr = *(int32 *)x_ptr ^ *(int32 *)y_ptr; break;
      case UINT32: *(uint32 *)z_ptr = *(uint32 *)x_ptr ^ *(uint32 *)y_ptr; break;
      case INT64: *(int64 *)z_ptr = *(int64 *)x_ptr ^ *(int64 *)y_ptr; break;
      case UINT64: *(uint64 *)z_ptr = *(uint64 *)x_ptr ^ *(uint64 *)y_ptr; break;
      case ERROR: {
        PyErr_SetString(PyExc_RuntimeError, "something is wrong i can feel it");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

// xnor
static PyObject *__xnor_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
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
      case INT8: *(int8 *)z_ptr = ~(*(int8 *)x_ptr ^ *(int8 *)y_ptr); break;
      case UINT8: *(uint8 *)z_ptr = ~(*(uint8 *)x_ptr ^ *(uint8 *)y_ptr); break;
      case INT16: *(int16 *)z_ptr = ~(*(int16 *)x_ptr ^ *(int16 *)y_ptr); break;
      case UINT16: *(uint16 *)z_ptr = ~(*(uint16 *)x_ptr ^ *(uint16 *)y_ptr); break;
      case INT32: *(int32 *)z_ptr = ~(*(int32 *)x_ptr ^ *(int32 *)y_ptr); break;
      case UINT32: *(uint32 *)z_ptr = ~(*(uint32 *)x_ptr ^ *(uint32 *)y_ptr); break;
      case INT64: *(int64 *)z_ptr = ~(*(int64 *)x_ptr ^ *(int64 *)y_ptr); break;
      case UINT64: *(uint64 *)z_ptr = ~(*(uint64 *)x_ptr ^ *(uint64 *)y_ptr); break;
      case ERROR: {
        PyErr_SetString(PyExc_RuntimeError, "something is wrong i can feel it");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *__sum_all__(const tensor_t *tx, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  char *px = (char *)tx->buf;
  size_t elem_size = tx->element_size;
  switch(dtype){
    case INT8: {
      *(int8 *)tz->buf = (int8)0;
      for(size_t i = 0; i < length; i++){
        char *x_ptr = px + i * elem_size;
        *(int8 *)tz->buf += *(int8 *)x_ptr;
      }
      return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
    }
    case UINT8: {
      *(uint8 *)tz->buf = (uint8)0;
      for(size_t i = 0; i < length; i++){
        char *x_ptr = px + i * elem_size;
        *(int8 *)tz->buf += *(uint8 *)x_ptr;
      }
      return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
    }
    case INT16: {
      *(int16 *)tz->buf = (int16)0;
      for(size_t i = 0; i < length; i++){
        char *x_ptr = px + i * elem_size;
        *(int16 *)tz->buf += *(int16 *)x_ptr;
      }
      return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
    }
    case UINT16: {
      *(uint16 *)tz->buf = (uint16)0;
      for(size_t i = 0; i < length; i++){
        char *x_ptr = px + i * elem_size;
        *(uint16 *)tz->buf += *(uint16 *)x_ptr;
      }
      return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
    }
    case INT32: {
      *(int32 *)tz->buf = (int32)0;
      for(size_t i = 0; i < length; i++){
        char *x_ptr = px + i * elem_size;
        *(int32* )tz->buf += *(int32 *)x_ptr;
      }
      return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
    }
    case UINT32: {
      *(uint32 *)tz->buf = (uint32)0;
      for(size_t i = 0; i < length; i++){
        char *x_ptr = px + i * elem_size;
        *(uint32 *)tz->buf += *(uint32 *)x_ptr;
      }
      return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
    }
    case INT64: {
      *(int64 *)tz->buf = (int64)0;
      for(size_t i = 0; i < length; i++){
        char *x_ptr = px + i * elem_size;
        *(int64 *)tz->buf += *(int64 *)x_ptr;
      }
      return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
    }
    case UINT64: {
      *(uint64 *)tz->buf = (uint64)0;
      for(size_t i = 0; i < length; i++){
        char *x_ptr = px + i * elem_size;
        *(uint64 *)tz->buf += *(uint64 *)x_ptr;
      }
      return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
    }
    case FP32: {
      *(float32 *)tz->buf = (float32)0;
      for(size_t i = 0; i < length; i++){
        char *x_ptr = px + i * elem_size;
        *(float32 *)tz->buf += *(float32 *)x_ptr;
      }
      return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
    }
    case FP64: {
      *(float64 *)tz->buf = (float64)0;
      for(size_t i = 0; i < length; i++){
        char *x_ptr = px + i * elem_size;
        *(float64 *)tz->buf += *(float64 *)x_ptr;
      }
      return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
    }
    case CMPX64: {
      (*(complex64 *)tz->buf).real = 0.0f;
      (*(complex64 *)tz->buf).imag = 0.0f;
      for(size_t i = 0; i < length; i++){
        char *x_ptr = px + i * elem_size;
        (*(complex64 *)tz->buf).real += (*(complex64 *)x_ptr).real;
        (*(complex64 *)tz->buf).imag += (*(complex64 *)x_ptr).imag;
      }
      return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
    }
    case CMPX128: {
      (*(complex128 *)tz->buf).real = 0.0;
      (*(complex128 *)tz->buf).imag = 0.0;
      for(size_t i = 0; i < length; i++){
        char *x_ptr = px + i * elem_size;
        (*(complex128 *)tz->buf).real += (*(complex128 *)x_ptr).real;
        (*(complex128 *)tz->buf).imag += (*(complex128 *)x_ptr).imag;
      }
      return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
    }
    default: {
      free(tz);
      PyErr_SetString(PyExc_RuntimeError, "something is wrong i can feel it");
      return NULL;
    }
  }
}

#define SUM_AXIS_KERNEL(NAME, IN_T, ACC_T)                            \
static PyObject *__sum_axis_##NAME##__(const tensor_t *tx, tensor_t *tz, int axis){\
  IN_T *in  = (IN_T*)tx->buf;                                         \
  IN_T *out = (IN_T*)tz->buf;                                         \
  int reduced_dim = tx->shape[axis];                                  \
  for(int out_index = 0; out_index < tz->size; out_index++){          \
    int tmp = out_index;                                              \
    int idx_in[16] = {0};                                             \
    int j = 0;                                                        \
    for(int i = 0; i < tx->ndim; i++){                                \
      if(i == axis) continue;                                         \
      idx_in[i] = tmp / tz->stride[j];                                \
      tmp %= tz->stride[j];                                           \
      j++;                                                            \
    }                                                                 \
    ACC_T total = 0;                                                  \
    for(int r = 0; r < reduced_dim; r++){                             \
      idx_in[axis] = r;                                               \
      int in_offset = 0;                                              \
      for(int k = 0; k < tx->ndim; k++){                              \
        in_offset += idx_in[k] * tx->stride[k];                       \
      }                                                               \
      total += (ACC_T)in[in_offset];                                  \
    }                                                                 \
    out[out_index] = (IN_T)total;                                     \
  }                                                                   \
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);     \
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

// mod
static PyObject *mod_(PyObject *self, PyObject *args){
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
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_Format(PyExc_TypeError, "%% opertaion on complex tensor is not supported");
  }
  if(tx->device.type != CPU || ty->device.type != CPU){
    PyErr_SetString(PyExc_RuntimeError, "Both tensor_t(s) must be on same device (CPU)");
    return NULL;
  }
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __mod_tensor__(tx, ty, tz);
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

// right shift
static PyObject *rshift(PyObject *self, PyObject *args){
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
  if(tx->dtype == FP32 || tx->dtype == FP64){
    PyErr_SetString(PyExc_RuntimeError, "'<<' operation on floating points tensor is not supported");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "'<<' operation on complex tensor is not supported");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_RuntimeError, "Internal error: size mismatch after broadcasting");
    return NULL;
  }
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __rshift_tensor__(tx, ty, tz);
}

// left shift
static PyObject *lshift(PyObject *self, PyObject *args){
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
  if(tx->dtype == FP32 || tx->dtype == FP64){
    PyErr_SetString(PyExc_RuntimeError, "'<<' operation on floating points tensor is not supported");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "'<<' operation on complex tensor is not supported");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_RuntimeError, "Internal error: size mismatch after broadcasting");
    return NULL;
  }
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __lshift_tensor__(tx, ty, tz);
}

static PyObject *and_(PyObject *self, PyObject *args){
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
  if(tx->dtype == FP32 || tx->dtype == FP64){
    PyErr_SetString(PyExc_RuntimeError, "'and' operation on floating points tensor is not supported");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "'and' operation on complex tensor is not supported");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_RuntimeError, "Internal error: size mismatch after broadcasting");
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
  if(tx->dtype == FP32 || tx->dtype == FP64){
    PyErr_SetString(PyExc_RuntimeError, "'nand' operation on floating points tensor is not supported");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "'nand' operation on complex tensor is not supported");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_RuntimeError, "Internal error: size mismatch after broadcasting");
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
  if(tx->dtype == FP32 || tx->dtype == FP64){
    PyErr_SetString(PyExc_RuntimeError, "'or' operation on floating points tensor is not supported");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "'or' operation on complex tensor is not supported");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_RuntimeError, "Internal error: size mismatch after broadcasting");
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
  if(tx->dtype == FP32 || tx->dtype == FP64){
    PyErr_SetString(PyExc_RuntimeError, "'nor' operation on floating points tensor is not supported");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "'nor' operation on complex tensor is not supported");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_RuntimeError, "Internal error: size mismatch after broadcasting");
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
    PyErr_SetString(PyExc_RuntimeError, "operands must be tensor capsules");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  if(tx->device.type != CPU){
    PyErr_SetString(PyExc_RuntimeError, "Only CPU tensors supported");
    return NULL;
  }
  if(tx->dtype == FP32 || tx->dtype == FP64){
    PyErr_SetString(PyExc_RuntimeError, "'not' operation on floating points tensor is not supported");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "'not' operation on complex tensor is not supported");
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
  if(tx->dtype == FP32 || tx->dtype == FP64){
    PyErr_SetString(PyExc_RuntimeError, "'xor' operation on floating points tensor is not supported");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "'xor' operation on complex tensor is not supported");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_RuntimeError, "Internal error: size mismatch after broadcasting");
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
  if(tx->dtype == FP32 || tx->dtype == FP64){
    PyErr_SetString(PyExc_RuntimeError, "'xnor' operation on floating points tensor is not supported");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "'xnor' operation on complex tensor is not supported");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_RuntimeError, "Internal error: size mismatch after broadcasting");
    return NULL;
  }
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __xnor_tensor__(tx, ty, tz);
}

static PyObject *permute(PyObject *self, PyObject *args){
  PyObject *x;
  PyObject *axes_tuple;
  if(!PyArg_ParseTuple(args, "OO", &x, &axes_tuple)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_TypeError, "expected tensor capsule");
    return NULL;
  }
  tensor_t *t = PyCapsule_GetPointer(x, "tensor_t on CPU");
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
  int *axes = malloc(sizeof(int) * ndim);
  bool *used = calloc(ndim, sizeof(bool));
  for(int i = 0; i < ndim; i++){
    int ax = (int)PyLong_AsLong(PyTuple_GetItem(axes_tuple, i));
    if(ax < 0) ax += ndim;
    if(ax < 0 || ax >= ndim){
      free(axes);
      free(used);
      PyErr_SetString(PyExc_IndexError, "axis out of range");
      return NULL;
    }
    if(used[ax]){
      free(axes);
      free(used);
      PyErr_SetString(PyExc_ValueError, "duplicate axis");
      return NULL;
    }
    used[ax] = 1;
    axes[i] = ax;
  }
  free(used);
  tensor_t *out = malloc(sizeof(tensor_t));
  out->dtype = t->dtype;
  out->ndim  = ndim;
  out->element_size = t->element_size;
  out->size  = t->size;
  out->shape  = malloc(sizeof(size_t) * ndim);
  out->stride = malloc(sizeof(size_t) * ndim);
  for(int i = 0; i < ndim; i++){
    out->shape[i] = t->shape[axes[i]];
  }
  out->stride[ndim - 1] = 1;
  for(int i = ndim - 2; i >= 0; i--){
    out->stride[i] = out->stride[i + 1] * out->shape[i + 1];
  }
  size_t itemsize = getsize(out->dtype);
  out->storage = malloc(sizeof(storage_t));
  out->storage->refcount = 1;
  out->storage->bytes = out->size * itemsize;
  out->storage->ptr = malloc(out->storage->bytes);
  out->buf = out->storage->ptr;
  size_t *out_index = malloc(sizeof(size_t) * ndim);
  for(size_t linear = 0; linear < out->size; linear++){
    size_t tmp = linear;
    for(int d = ndim - 1; d >= 0; d--){
      out_index[d] = tmp % out->shape[d];
      tmp /= out->shape[d];
    }
    size_t in_offset = 0;
    for(int d = 0; d < ndim; d++){
      int orig_axis = axes[d];
      in_offset += out_index[d] * t->stride[orig_axis];
    }
    memcpy(
      (char*)out->buf + linear * itemsize,
      (char*)t->buf   + in_offset * itemsize,
      itemsize
    );
  }
  free(out_index);
  free(axes);
  return PyCapsule_New(out, "tensor_t on CPU", capsule_destroyer);
}

SUM_AXIS_KERNEL(int8, int8, int32);
SUM_AXIS_KERNEL(uint8, uint8, uint64);
SUM_AXIS_KERNEL(int16, int16, int32);
SUM_AXIS_KERNEL(uint16, uint16, uint64);
SUM_AXIS_KERNEL(int32, int32, int64);
SUM_AXIS_KERNEL(uint32, uint32, uint64);
SUM_AXIS_KERNEL(int64, int64, int64);
SUM_AXIS_KERNEL(uint64, uint64, uint64);
SUM_AXIS_KERNEL(float32, float32, float32);
SUM_AXIS_KERNEL(float64, float64, float64);

static PyObject *__sum_axis_cmpx64__(tensor_t *tx, tensor_t *tz, int axis){
  complex64 *in = (complex64 *)tx->buf;
  complex64 *out = (complex64 *)tz->buf;
  int reduced_dim = tx->shape[axis];
  for(int out_index = 0; out_index < tz->size; out_index++){
    int tmp = out_index;
    int idx_in[16] = {0};
    int j = 0;
    for(int i = 0; i < tx->ndim; i++){
      if(i == axis) continue;
      idx_in[i] = tmp / tz->stride[j];
      tmp %= tz->stride[j];
      j++;
    }
    float real_sum = 0.0f;
    float imag_sum = 0.0f;
    for(int r = 0; r < reduced_dim; r++){
      idx_in[axis] = r;
      int in_offset = 0;
      for(int k = 0; k < tx->ndim; k++){
        in_offset += idx_in[k] * tx->stride[k];
      }
      real_sum += (float)in[in_offset].real;
      imag_sum += (float)in[in_offset].imag;
    }
    out[out_index].real = (float)real_sum;
    out[out_index].imag = (float)imag_sum;
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *__sum_axis_cmpx128__(tensor_t *tx, tensor_t *tz, int axis){
  complex128 *in = (complex128 *)tx->buf;
  complex128 *out = (complex128 *)tz->buf;
  int reduced_dim = tx->shape[axis];
  for(int out_index = 0; out_index < tz->size; out_index++){
    int tmp = out_index;
    int idx_in[16] = {0};
    int j = 0;
    for(int i = 0; i < tx->ndim; i++){
      if(i == axis) continue;
      idx_in[i] = tmp / tz->stride[j];
      tmp %= tz->stride[j];
      j++;
    }
    double real_sum = 0.0;
    double imag_sum = 0.0;
    for(int r = 0; r < reduced_dim; r++){
      idx_in[axis] = r;
      int in_offset = 0;
      for(int k = 0; k < tx->ndim; k++){
        in_offset += idx_in[k] * tx->stride[k];
      }
      real_sum += (double)in[in_offset].real;
      imag_sum += (double)in[in_offset].imag;
    }
    out[out_index].real = (double)real_sum;
    out[out_index].imag = (double)imag_sum;
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *sum(PyObject *self, PyObject *args, PyObject *kwargs){
  PyObject *x;
  PyObject *axis_obj = Py_None;
  static char *kwlist[] = {"x", "axis", NULL};
  if(!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist, &x, &axis_obj)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid capsule");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor pointer");
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
    PyErr_SetString(PyExc_ValueError, "axis is out of range");
    return NULL;
  }
  int out_ndim = tx->ndim - 1;
  if(out_ndim == 0){
    tensor_t *tz = tensor_empty_scalar_like(tx);
    if(!tz){
      PyErr_SetString(PyExc_RuntimeError, "scalar tensor allocation failed");
      return NULL;
    }
    switch(tx->dtype){
      case INT8: return __sum_axis_int8__(tx, tz, axis);
      case UINT8: return __sum_axis_uint8__(tx, tz, axis);
      case INT16: return __sum_axis_int16__(tx, tz, axis);
      case UINT16: return __sum_axis_uint16__(tx, tz, axis);
      case INT32: return __sum_axis_int32__(tx, tz, axis);
      case UINT32: return __sum_axis_uint32__(tx, tz, axis);
      case INT64: return __sum_axis_int64__(tx, tz, axis);
      case UINT64: return __sum_axis_uint64__(tx, tz, axis);
      case FP32: return __sum_axis_float32__(tx, tz, axis);
      case FP64: return __sum_axis_float64__(tx, tz, axis);
      case CMPX64: return __sum_axis_cmpx64__(tx, tz, axis);
      case CMPX128: return __sum_axis_cmpx128__(tx, tz, axis);
      case ERROR: {
        destroy(tz);
        PyErr_SetString(PyExc_RuntimeError, "something is wrong i can feel it");
        return NULL;
      }
      default: {
        destroy(tz);
        PyErr_SetString(PyExc_TypeError, "sum not supported for this dtype");
        return NULL;
      }
    }
  }
  tensor_t *tz = malloc(sizeof(tensor_t));
  if(!tz){
    PyErr_SetString(PyExc_MemoryError, "tensor allocation failed");
    return NULL;
  }
  tz->dtype = tx->dtype;
  tz->device = tx->device;
  tz->element_size = tx->element_size;
  tz->ndim = out_ndim;
  tz->shape = malloc(sizeof(size_t) * out_ndim);
  tz->stride = malloc(sizeof(size_t) * out_ndim);
  if(!tz->shape || !tz->stride){
    PyErr_SetString(PyExc_MemoryError, "shape/stride allocation failed");
    return NULL;
  }
  int j = 0;
  for(int i = 0; i < tx->ndim; i++){
    if(i == axis) continue;
    tz->shape[j++] = tx->shape[i];
  }
  tz->stride[out_ndim - 1] = 1;
  for(int i = out_ndim - 2; i >= 0; i--){
    tz->stride[i] = tz->shape[i + 1] * tz->stride[i + 1];
  }
  tz->size = 1;
  for(int i = 0; i < out_ndim; i++){
    tz->size *= tz->shape[i];
  }
  tz->storage = malloc(sizeof(storage_t));
  tz->storage->bytes = tz->size * tz->element_size;
  tz->storage->device = tz->device;
  tz->storage->refcount = 1;
  tz->storage->ptr = calloc(1, tz->storage->bytes);
  if(!tz->storage->ptr){
    PyErr_SetString(PyExc_RuntimeError, "storage allocation failed");
    return NULL;
  }
  tz->buf = tz->storage->ptr;
  switch(tx->dtype){
    case INT8: return __sum_axis_int8__(tx, tz, axis);
    case UINT8: return __sum_axis_uint8__(tx, tz, axis);
    case INT16: return __sum_axis_int16__(tx, tz, axis);
    case UINT16: return __sum_axis_uint16__(tx, tz, axis);
    case INT32: return __sum_axis_int32__(tx, tz, axis);
    case UINT32: return __sum_axis_uint32__(tx, tz, axis);
    case INT64: return __sum_axis_int64__(tx, tz, axis);
    case UINT64: return __sum_axis_uint64__(tx, tz, axis);
    case FP32: return __sum_axis_float32__(tx, tz, axis);
    case FP64: return __sum_axis_float64__(tx, tz, axis);
    case CMPX64: return __sum_axis_cmpx64__(tx, tz, axis);
    case CMPX128: return __sum_axis_cmpx128__(tx, tz, axis);
    case ERROR: {
      destroy(tz);
      PyErr_SetString(PyExc_RuntimeError, "something is wrong i can feel it");
      return NULL;
    }
    default: {
      destroy(tz);
      PyErr_SetString(PyExc_TypeError, "sum not supported for this dtype");
      return NULL;
    }
  }
}

static PyObject *bmm(PyObject *self, PyObject *args){
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
  if(tx->ndim < 2 || ty->ndim < 2){
    PyErr_SetString(PyExc_RuntimeError, "Both tensors must be at least 2D for matmul");
    return NULL;
  }
  if(tx->ndim != ty->ndim){
    PyErr_SetString(PyExc_ValueError, "Both tensors must have same number of dimensions for matmul");
    return NULL;
  }
  size_t m = tx->shape[tx->ndim - 2];
  size_t k = tx->shape[tx->ndim - 1];
  size_t k2 = ty->shape[ty->ndim - 2];
  size_t n = ty->shape[ty->ndim - 1];
  if(k != k2){
    PyErr_SetString(PyExc_ValueError, "Matrix dimensions incompatible for multiplication");
    return NULL;
  }
  for(size_t i = 0; i < tx->ndim - 2; i++){
    if(tx->shape[i] != ty->shape[i]){
      PyErr_SetString(PyExc_ValueError, "Batch dimensions must match");
      return NULL;
    }
  }
  tensor_t *tz = malloc(sizeof(tensor_t));
  if(!tz){
    PyErr_SetString(PyExc_MemoryError, "tensor allocation failed");
    return NULL;
  }
  tz->dtype = tx->dtype;
  tz->device = tx->device;
  tz->element_size = tx->element_size;
  tz->ndim = tx->ndim;
  tz->shape = malloc(sizeof(size_t) * tz->ndim);
  if(!tz->shape){
    free(tz);
    PyErr_SetString(PyExc_MemoryError, "shape allocation failed");
    return NULL;
  }
  for(size_t i = 0; i < tz->ndim - 2; i++){
    tz->shape[i] = tx->shape[i];
  }
  tz->shape[tz->ndim - 2] = m;
  tz->shape[tz->ndim - 1] = n;
  tz->stride = malloc(sizeof(size_t) * tz->ndim);
  if(!tz->stride){
    free(tz->shape);
    free(tz);
    PyErr_SetString(PyExc_MemoryError, "stride allocation failed");
    return NULL;
  }
  tz->stride[tz->ndim - 1] = 1;
  for(int i = tz->ndim - 2; i >= 0; i--){
    tz->stride[i] = tz->shape[i + 1] * tz->stride[i + 1];
  }
  tz->size = 1;
  for(size_t i = 0; i < tz->ndim; i++){
    tz->size *= tz->shape[i];
  }
  tz->storage = malloc(sizeof(storage_t));
  if(!tz->storage){
    free(tz->stride);
    free(tz->shape);
    free(tz);
    PyErr_SetString(PyExc_MemoryError, "storage allocation failed");
    return NULL;
  }
  tz->storage->bytes = tz->size * tz->element_size;
  tz->storage->device = tz->device;
  tz->storage->refcount = 1;
  tz->storage->ptr = calloc(1, tz->storage->bytes);
  if(!tz->storage->ptr){
    free(tz->storage);
    free(tz->stride);
    free(tz->shape);
    free(tz);
    PyErr_SetString(PyExc_MemoryError, "storage.ptr allocation failed");
    return NULL;
  }
  tz->buf = tz->storage->ptr;
  switch(tx->dtype){
    case INT8: {
      int8 *a = (int8 *)tx->buf;
      int8 *b = (int8 *)ty->buf;
      int8 *c = (int8 *)tz->buf;
      size_t batch_size = 1;
      for(size_t i = 0; i < tx->ndim - 2; i++){
        batch_size *= tx->shape[i];
      }
      for(size_t batch = 0; batch < batch_size; batch++){
        size_t a_offset = batch * m * k;
        size_t b_offset = batch * k * n;
        size_t c_offset = batch * m * n;
        for(size_t i = 0; i < m; i++){
          for(size_t kk = 0; kk < k; kk++){
            int8 a_val = a[a_offset + i * k +kk];
            for(size_t j = 0; j < n; j++){
              c[c_offset + i * n +j] += a_val * b[b_offset + kk * n + j];
            }
          }
        }
      }
      break;
    }
    case UINT8: {
      uint8 *a = (uint8 *)tx->buf;
      uint8 *b = (uint8 *)ty->buf;
      uint8 *c = (uint8 *)tz->buf;
      size_t batch_size = 1;
      for(size_t i = 0; i < tx->ndim - 2; i++){
        batch_size *= tx->shape[i];
      }
      for(size_t batch = 0; batch < batch_size; batch++){
        size_t a_offset = batch * m * k;
        size_t b_offset = batch * k * n;
        size_t c_offset = batch * m * n;
        for(size_t i = 0; i < m; i++){
          for(size_t kk = 0; kk < k; kk++){
            uint8 a_val = a[a_offset + i * k +kk];
            for(size_t j = 0; j < n; j++){
              c[c_offset + i * n +j] += a_val * b[b_offset + kk * n + j];
            }
          }
        }
      }
      break;
    }
    case INT16: {
      int16 *a = (int16 *)tx->buf;
      int16 *b = (int16 *)ty->buf;
      int16 *c = (int16 *)tz->buf;
      size_t batch_size = 1;
      for(size_t i = 0; i < tx->ndim - 2; i++){
        batch_size *= tx->shape[i];
      }
      for(size_t batch = 0; batch < batch_size; batch++){
        size_t a_offset = batch * m * k;
        size_t b_offset = batch * k * n;
        size_t c_offset = batch * m * n;
        for(size_t i = 0; i < m; i++){
          for(size_t kk = 0; kk < k; kk++){
            int16 a_val = a[a_offset + i * k +kk];
            for(size_t j = 0; j < n; j++){
              c[c_offset + i * n +j] += a_val * b[b_offset + kk * n + j];
            }
          }
        }
      }
      break;
    }
    case UINT16: {
      uint16 *a = (uint16 *)tx->buf;
      uint16 *b = (uint16 *)ty->buf;
      uint16 *c = (uint16 *)tz->buf;
      size_t batch_size = 1;
      for(size_t i = 0; i < tx->ndim - 2; i++){
        batch_size *= tx->shape[i];
      }
      for(size_t batch = 0; batch < batch_size; batch++){
        size_t a_offset = batch * m * k;
        size_t b_offset = batch * k * n;
        size_t c_offset = batch * m * n;
        for(size_t i = 0; i < m; i++){
          for(size_t kk = 0; kk < k; kk++){
            uint16 a_val = a[a_offset + i * k +kk];
            for(size_t j = 0; j < n; j++){
              c[c_offset + i * n +j] += a_val * b[b_offset + kk * n + j];
            }
          }
        }
      }
      break;
    }
    case INT32: {
      int32 *a = (int32 *)tx->buf;
      int32 *b = (int32 *)ty->buf;
      int32 *c = (int32 *)tz->buf;
      size_t batch_size = 1;
      for(size_t i = 0; i < tx->ndim - 2; i++){
        batch_size *= tx->shape[i];
      }
      for(size_t batch = 0; batch < batch_size; batch++){
        size_t a_offset = batch * m * k;
        size_t b_offset = batch * k * n;
        size_t c_offset = batch * m * n;
        for(size_t i = 0; i < m; i++){
          for(size_t kk = 0; kk < k; kk++){
            int32 a_val = a[a_offset + i * k + kk];
            for(size_t j = 0; j < n; j++){
              c[c_offset + i * n + j] += a_val * b[b_offset + kk * n + j];
            }
          }
        }
      }
      break;
    }
    case UINT32: {
      uint32 *a = (uint32 *)tx->buf;
      uint32 *b = (uint32 *)ty->buf;
      uint32 *c = (uint32 *)tz->buf;
      size_t batch_size = 1;
      for(size_t i = 0; i < tx->ndim - 2; i++){
        batch_size *= tx->shape[i];
      }
      for(size_t batch = 0; batch < batch_size; batch++){
        size_t a_offset = batch * m * k;
        size_t b_offset = batch * k * n;
        size_t c_offset = batch * m * n;
        for(size_t i = 0; i < m; i++){
          for(size_t kk = 0; kk < k; kk++){
            uint32 a_val = a[a_offset + i * k + kk];
            for(size_t j = 0; j < n; j++){
              c[c_offset + i * n + j] += a_val * b[b_offset + kk * n + j];
            }
          }
        }
      }
      break;
    }
    case INT64: {
      int64 *a = (int64 *)tx->buf;
      int64 *b = (int64 *)ty->buf;
      int64 *c = (int64 *)tz->buf;
      size_t batch_size = 1;
      for(size_t i = 0; i < tx->ndim - 2; i++){
        batch_size *= tx->shape[i];
      }
      for(size_t batch = 0; batch < batch_size; batch++){
        size_t a_offset = batch * m * k;
        size_t b_offset = batch * k * n;
        size_t c_offset = batch * m * n;
        for(size_t i = 0; i < m; i++){
          for(size_t kk = 0; kk < k; kk++){
            int64 a_val = a[a_offset + i * k + kk];
            for(size_t j = 0; j < n; j++){
              c[c_offset + i * n + j] += a_val * b[b_offset + kk * n + j];
            }
          }
        }
      }
      break;
    }
    case UINT64: {
      uint64 *a = (uint64 *)tx->buf;
      uint64 *b = (uint64 *)ty->buf;
      uint64 *c = (uint64 *)tz->buf;
      size_t batch_size = 1;
      for(size_t i = 0; i < tx->ndim - 2; i++){
        batch_size *= tx->shape[i];
      }
      for(size_t batch = 0; batch < batch_size; batch++){
        size_t a_offset = batch * m * k;
        size_t b_offset = batch * k * n;
        size_t c_offset = batch * m * n;
        for(size_t i = 0; i < m; i++){
          for(size_t kk = 0; kk < k; kk++){
            uint64 a_val = a[a_offset + i * k + kk];
            for(size_t j = 0; j < n; j++){
              c[c_offset + i * n + j] += a_val * b[b_offset + kk * n + j];
            }
          }
        }
      }
      break;
    }
    case FP32: {
      float32 *a = (float32 *)tx->buf;
      float32 *b = (float32 *)ty->buf;
      float32 *c = (float32 *)tz->buf;
      size_t batch_size = 1;
      for(size_t i = 0; i < tx->ndim - 2; i++){
        batch_size *= tx->shape[i];
      }
      for(size_t batch = 0; batch < batch_size; batch++){
        size_t a_offset = batch * m * k;
        size_t b_offset = batch * k * n;
        size_t c_offset = batch * m * n;
        for(size_t i = 0; i < m; i++){
          for(size_t kk = 0; kk < k; kk++){
            float32 a_val = a[a_offset + i * k + kk];
            for(size_t j = 0; j < n; j++){
              c[c_offset + i * n + j] += a_val * b[b_offset + kk * n + j];
            }
          }
        }
      }
      break;
    }
    case FP64: {
      float64 *a = (float64 *)tx->buf;
      float64 *b = (float64 *)ty->buf;
      float64 *c = (float64 *)tz->buf;
      size_t batch_size = 1;
      for(size_t i = 0; i < tx->ndim - 2; i++){
        batch_size *= tx->shape[i];
      }
      for(size_t batch = 0; batch < batch_size; batch++){
        size_t a_offset = batch * m * k;
        size_t b_offset = batch * k * n;
        size_t c_offset = batch * m * n;
        for(size_t i = 0; i < m; i++){
          for(size_t kk = 0; kk < k; kk++){
            float64 a_val = a[a_offset + i * k + kk];
            for(size_t j = 0; j < n; j++){
              c[c_offset + i * n + j] += a_val * b[b_offset + kk * n + j];
            }
          }
        }
      }
      break;
    }
    case CMPX64: {
      complex64 *a = (complex64 *)tx->buf;
      complex64 *b = (complex64 *)ty->buf;
      complex64 *c = (complex64 *)tz->buf;
      size_t batch_size = 1;
      for(size_t i = 0; i < tx->ndim - 2; i++){
        batch_size *= tx->shape[i];
      }
      for(size_t batch = 0; batch < batch_size; batch++){
        size_t a_offset = batch * m * k;
        size_t b_offset = batch * k * n;
        size_t c_offset = batch * m * n;
        for(size_t i = 0; i < m; i++){
          for(size_t kk = 0; kk < k; kk++){
            complex64 a_val = a[a_offset + i * k + kk];
            for(size_t j = 0; j < n; j++){
              size_t c_idx = c_offset + i * n + j;
              size_t b_idx = b_offset + kk * n + j;
              float32 real_part = a_val.real * b[b_idx].real - a_val.imag * b[b_idx].imag;
              float32 imag_part = a_val.real * b[b_idx].imag + a_val.imag * b[b_idx].real;
              c[c_idx].real += real_part;
              c[c_idx].imag += imag_part;
            }
          }
        }
      }
      break;
    }
    case CMPX128: {
      complex128 *a = (complex128 *)tx->buf;
      complex128 *b = (complex128 *)ty->buf;
      complex128 *c = (complex128 *)tz->buf;
      size_t batch_size = 1;
      for(size_t i = 0; i < tx->ndim - 2; i++){
        batch_size *= tx->shape[i];
      }
      for(size_t batch = 0; batch < batch_size; batch++){
        size_t a_offset = batch * m * k;
        size_t b_offset = batch * k * n;
        size_t c_offset = batch * m * n;
        for(size_t i = 0; i < m; i++){
          for(size_t kk = 0; kk < k; kk++){
            complex128 a_val = a[a_offset + i * k + kk];
            for(size_t j = 0; j < n; j++){
              size_t c_idx = c_offset + i * n + j;
              size_t b_idx = b_offset + kk * n + j;
              float64 real_part = a_val.real * b[b_idx].real - a_val.imag * b[b_idx].imag;
              float64 imag_part = a_val.real * b[b_idx].imag + a_val.imag * b[b_idx].real;
              c[c_idx].real += real_part;
              c[c_idx].imag += imag_part;
            }
          }
        }
      }
      break;
    }
    default: {
      destroy(tz);
      PyErr_SetString(PyExc_TypeError, "matmul not supported for this dtype");
      return NULL;
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *real(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "invalid tensor capsule");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  if(tx->dtype != CMPX64 && tx->dtype != CMPX128){
    PyErr_SetString(PyExc_TypeError, "real() implemented only for complex dtype tensor(s)");
    return NULL;
  }
  tensor_t *tz = malloc(sizeof(tensor_t));
  tz->dtype = (tx->dtype == CMPX64) ? FP32 : FP64;
  tz->element_size = getsize(tz->dtype);
  tz->device = (device_t){CPU, 0};
  tz->size = 1;
  tz->storage = malloc(sizeof(storage_t));
  tz->storage->device = tz->device;
  tz->storage->refcount = 1;
  tz->storage->bytes = tz->element_size;
  tz->storage->ptr = malloc(tz->storage->bytes);
  tz->buf = tz->storage->ptr;
  if(tx->ndim == 0){
    tz->ndim = 0;
    tz->shape = NULL;
    tz->stride = NULL;
    if(tx->dtype == CMPX64){
      ((float*)tz->buf)[0] = ((complex64*)tx->buf)[0].real;
    }else{
      ((double*)tz->buf)[0] = ((complex128*)tx->buf)[0].real;
    }
    return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
  }
  tz->ndim = tx->ndim;
  tz->shape = malloc(sizeof(size_t) * tz->ndim);
  tz->stride = malloc(sizeof(size_t) * tz->ndim);
  for(size_t i = 0; i < tz->ndim; i++){
    tz->shape[i] = tx->shape[i];
    tz->stride[i] = tx->stride[i];
  }
  tz->size = tx->size;
  tz->storage->bytes = tz->size * tz->element_size;
  tz->storage->ptr = malloc(tz->storage->bytes);
  tz->buf = tz->storage->ptr;
  if(tx->dtype == CMPX64){
    complex64 *src = (complex64*)tx->buf;
    float *dst = (float*)tz->buf;
    for(size_t i = 0; i < tx->size; i++) dst[i] = src[i].real;
  }else{
    complex128 *src = (complex128*)tx->buf;
    double *dst = (double*)tz->buf;
    for(size_t i = 0; i < tx->size; i++) dst[i] = src[i].real;
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *imag(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "invalid tensor capsule");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  if(tx->dtype != CMPX64 && tx->dtype != CMPX128){
    PyErr_SetString(PyExc_TypeError, "imag() implemented only for complex dtype tensor(s)");
    return NULL;
  }
  tensor_t *tz = malloc(sizeof(tensor_t));
  tz->dtype = (tx->dtype == CMPX64) ? FP32 : FP64;
  tz->element_size = getsize(tz->dtype);
  tz->device = (device_t){CPU, 0};
  tz->size = 1;
  tz->storage = malloc(sizeof(storage_t));
  tz->storage->device = tz->device;
  tz->storage->refcount = 1;
  tz->storage->bytes = tz->element_size;
  tz->storage->ptr = malloc(tz->storage->bytes);
  tz->buf = tz->storage->ptr;
  if(tx->ndim == 0){
    tz->ndim = 0;
    tz->shape = NULL;
    tz->stride = NULL;
    if(tx->dtype == CMPX64){
      ((float*)tz->buf)[0] = ((complex64*)tx->buf)[0].imag;
    }else{
      ((double*)tz->buf)[0] = ((complex128*)tx->buf)[0].imag;
    }
    return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
  }
  tz->ndim = tx->ndim;
  tz->shape = malloc(sizeof(size_t) * tz->ndim);
  tz->stride = malloc(sizeof(size_t) * tz->ndim);
  for(size_t i = 0; i < tz->ndim; i++){
    tz->shape[i] = tx->shape[i];
    tz->stride[i] = tx->stride[i];
  }
  tz->size = tx->size;
  tz->storage->bytes = tz->size * tz->element_size;
  tz->storage->ptr = malloc(tz->storage->bytes);
  tz->buf = tz->storage->ptr;
  if(tx->dtype == CMPX64){
    complex64 *src = (complex64*)tx->buf;
    float *dst = (float*)tz->buf;
    for(size_t i = 0; i < tx->size; i++) dst[i] = src[i].imag;
  }else{
    complex128 *src = (complex128*)tx->buf;
    double *dst = (double*)tz->buf;
    for(size_t i = 0; i < tx->size; i++) dst[i] = src[i].imag;
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyMethodDef methods[] = {
  {"add", add, METH_VARARGS, "element-wise 'add' operation on tensor"},
  {"sub", sub, METH_VARARGS, "element-wise 'sub' operation tensor"},
  {"mul", mul, METH_VARARGS, "element-wise 'mul' operation on tensor"},
  {"tdiv", tdiv, METH_VARARGS, "element-wise 'tdiv' operation tensor"},
  {"fdiv", fdiv_, METH_VARARGS, "element-wise 'fdiv' operation on tensor"},
  {"pow", pow_, METH_VARARGS, "element-wise 'pow' operation on tensor"},
  {"mod", mod_, METH_VARARGS, "element-wise 'mod' operation on tensor"},
  {"eq", eq, METH_VARARGS, "element-wise 'eq' operation on tensor"},
  {"ne", ne, METH_VARARGS, "element-wise 'neq' operation on tensor"},
  {"gt", gt, METH_VARARGS, "element-wise 'gt' operation on tensor"},
  {"ge", ge, METH_VARARGS, "element-wise 'ge' operation on tensor"},
  {"lt", lt, METH_VARARGS, "element-wise 'lt' operation on tensor"},
  {"le", le, METH_VARARGS, "element-wise 'le' operation on tensor"},
  {"neg", neg, METH_VARARGS, "element-wise 'neg' operation on tensor"},
  {"pos", pos, METH_VARARGS, "element-wise 'pos' operation on tensor"},
  {"abs", abs_, METH_VARARGS, "element-wise 'abs' operation on tensor"},
  {"rshift", rshift, METH_VARARGS, "element-wise 'rshift' operation on tensor"},
  {"lshift", lshift, METH_VARARGS, "element-wise 'lshift' operation on tensor"},
  {"and_", and_, METH_VARARGS, "element-wise 'and' operation on tensor"},
  {"nand_", nand_, METH_VARARGS, "element-wise 'nand' operation on tensor"},
  {"or_", or_, METH_VARARGS, "element-wise 'or' operation on tensor"},
  {"nor_", nor_, METH_VARARGS, "element-wise 'nor' operation on tensor"},
  {"not_", not_, METH_VARARGS, "element-wise 'not' operation on tensor"},
  {"xor_", xor_, METH_VARARGS, "element-wise 'xor' operation on tensor"},
  {"xnor_", xnor_, METH_VARARGS, "element-wise 'xnor' operation on tensor"},
  {"permute", permute, METH_VARARGS, "tensor permute"},
  {"sum", (PyCFunction)sum, METH_VARARGS | METH_KEYWORDS, "returns the sum of the tensor"},
  {"bmm", bmm, METH_VARARGS, "compute batch matrix multiplication on tensor"},
  {"real", real, METH_VARARGS, "get real values from complex tensor"},
  {"imag", imag, METH_VARARGS, "get imag values from complex tensor"},
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
