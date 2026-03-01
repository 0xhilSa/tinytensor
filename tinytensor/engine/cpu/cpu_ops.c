#include <python3.10/Python.h>
#include "../tt_memory.h"
#include "../tensor.h"

void capsule_destroyer(PyObject *capsule){
  tensor_t *t = PyCapsule_GetPointer(capsule, "tensor_t on CPU");
  if(t){
    destroy(t);
  }
}

tensor_t *tensor_empty_scalar_like(tensor_t *tx){
  tensor_t *tz = tt_malloc(sizeof(tensor_t));
  if(!tz) return NULL;
  tz->dtype = tx->dtype;
  tz->device = tx->device;
  tz->element_size = tx->element_size;
  tz->ndim = 0;
  tz->size = 1;
  tz->shape = NULL;
  tz->stride = NULL;
  tz->storage = tt_malloc(sizeof(storage_t));
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
  tensor_t *tz = tt_malloc(sizeof(tensor_t));
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
    tz->shape = tt_malloc(sizeof(size_t) * tz->ndim);
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
  tz->storage = tt_malloc(sizeof(storage_t));
  if(!tz->storage){
    if(tz->shape) free(tz->shape);
    free(tz);
    PyErr_SetString(PyExc_RuntimeError, "tensor_t.storage allocation failed!");
    return NULL;
  }
  tz->storage->bytes = tz->size * tz->element_size;
  tz->storage->device = tz->device;
  tz->storage->refcount = 1;
  tz->storage->ptr = tt_malloc(tz->storage->bytes);
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
      case FP16: {
        float16 hx = *(float16 *)x_ptr;
        float16 hy = *(float16 *)y_ptr;
        float32 fx = fp16_to_float(hx);
        float32 fy = fp16_to_float(hy);
        float32 fz = fx + fy;
        *(float16 *)z_ptr = float_to_fp16(fz);
        break;
      }
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
        PyErr_SetString(PyExc_RuntimeError, "something went wrong");
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
      case FP16: {
        float16 hx = *(float16 *)x_ptr;
        float16 hy = *(float16 *)y_ptr;
        float32 fx = fp16_to_float(hx);
        float32 fy = fp16_to_float(hy);
        float32 fz = fx - fy;
        *(float16 *)z_ptr = float_to_fp16(fz);
        break;
      }
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
        PyErr_SetString(PyExc_RuntimeError, "something went wrong");
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
      case FP16: {
        float16 hx = *(float16 *)x_ptr;
        float16 hy = *(float16 *)y_ptr;
        float32 fx = fp16_to_float(hx);
        float32 fy = fp16_to_float(hy);
        float32 fz = fx * fy;
        *(float16 *)z_ptr = float_to_fp16(fz);
        break;
      }
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
        PyErr_SetString(PyExc_RuntimeError, "something went wrong");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *__tdiv_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t in_dtype  = tx->dtype;
  dtype_t out_dtype = tz->dtype;
  char *px = (char *)tx->buf;
  char *py = (char *)ty->buf;
  char *pz = (char *)tz->buf;
  size_t in_elem_size  = tx->element_size;
  size_t out_elem_size = tz->element_size;
  for (size_t i = 0; i < length; i++) {
    char *x_ptr = px + i * in_elem_size;
    char *y_ptr = py + i * in_elem_size;
    char *z_ptr = pz + i * out_elem_size;
    float64 fx = 0.0;
    float64 fy = 0.0;
    switch (in_dtype){
      case BOOL:
        fx = (float64)(*(bool *)x_ptr);
        fy = (float64)(*(bool *)y_ptr);
        break;
      case INT8:
        fx = (float64)(*(int8 *)x_ptr);
        fy = (float64)(*(int8 *)y_ptr);
        break;
      case UINT8:
        fx = (float64)(*(uint8 *)x_ptr);
        fy = (float64)(*(uint8 *)y_ptr);
        break;
      case INT16:
        fx = (float64)(*(int16 *)x_ptr);
        fy = (float64)(*(int16 *)y_ptr);
        break;
      case UINT16:
        fx = (float64)(*(uint16 *)x_ptr);
        fy = (float64)(*(uint16 *)y_ptr);
        break;
      case INT32:
        fx = (float64)(*(int32 *)x_ptr);
        fy = (float64)(*(int32 *)y_ptr);
        break;
      case UINT32:
        fx = (float64)(*(uint32 *)x_ptr);
        fy = (float64)(*(uint32 *)y_ptr);
        break;
      case INT64:
        fx = (float64)(*(int64 *)x_ptr);
        fy = (float64)(*(int64 *)y_ptr);
        break;
      case UINT64:
        fx = (float64)(*(uint64 *)x_ptr);
        fy = (float64)(*(uint64 *)y_ptr);
        break;
      case FP16:
        fx = (float64)fp16_to_float(*(float16 *)x_ptr);
        fy = (float64)fp16_to_float(*(float16 *)y_ptr);
        break;
      case FP32:
        fx = (float64)(*(float32 *)x_ptr);
        fy = (float64)(*(float32 *)y_ptr);
        break;
      case FP64:
        fx = *(float64 *)x_ptr;
        fy = *(float64 *)y_ptr;
        break;
      default:
        PyErr_SetString(PyExc_TypeError, "Unsupported dtype for division");
        return NULL;
    }
    float64 fz = fx / fy;
    switch (out_dtype){
      case FP16: *(float16 *)z_ptr = float_to_fp16((float)fz); break;
      case FP32: *(float32 *)z_ptr = (float32)fz; break;
      case FP64: *(float64 *)z_ptr = (float64)fz; break;
      default: PyErr_SetString(PyExc_TypeError, "Invalid output dtype for division"); return NULL;
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
        float32 a = x.real;
        float32 b = x.imag;
        float32 c = y.real;
        float32 d = y.imag;
        float32 denom = c*c + d*d;
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
        float64 a = x.real;
        float64 b = x.imag;
        float64 c = y.real;
        float64 d = y.imag;
        float64 denom = c*c + d*d;
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
        PyErr_SetString(PyExc_RuntimeError, "something went wrong");
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
      case FP16: {
        float16_t hx = *(float16_t *)x_ptr;
        float16_t hy = *(float16_t *)y_ptr;
        float32 fx = fp16_to_float(hx);
        float32 fy = fp16_to_float(hy);
        *(int64 *)z_ptr = (int64)(fx / fy);
        break;
      }
      case FP32: *(int64 *)z_ptr = (int64)(*(float32 *)x_ptr / *(float32 *)y_ptr); break;
      case FP64: *(int64 *)z_ptr = (int64)(*(float64 *)x_ptr / *(float64 *)y_ptr); break;
      case ERROR: {
        PyErr_SetString(PyExc_RuntimeError, "something went wrong");
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
      case FP16: {
        float16_t hx = *(float16_t *)x_ptr;
        float16_t hy = *(float16_t *)y_ptr;
        float32 fx = fp16_to_float(hx);
        float32 fy = fp16_to_float(hy);
        float32 fz = powf(fx, fy);
        *(float16_t *)z_ptr = float_to_fp16(fz);
        break;
      }
      case FP32: *(float32 *)z_ptr = (float32)powf(*(float32 *)x_ptr, *(float32 *)y_ptr); break;
      case FP64: *(float64 *)z_ptr = (float64)pow(*(float64 *)x_ptr, *(float64 *)y_ptr); break;
      case CMPX64: {
        complex64 x = *(complex64 *)x_ptr;
        complex64 y = *(complex64 *)y_ptr;
        float32 r = sqrt(x.real * x.real + x.imag * x.imag);
        float32 theta = atan(x.imag/x.real);
        float32 A = y.real * log(r) - y.imag * theta;
        float32 B = y.imag * log(r) + y.real * theta;
        float32 e = exp(A);
        complex64 z = {cos(B), sin(B)};
        (*(complex64 *)z_ptr).real = e * z.real;
        (*(complex64 *)z_ptr).imag = e * z.imag;
        break;
      }
      case CMPX128: {
        complex128 x = *(complex128 *)x_ptr;
        complex128 y = *(complex128 *)y_ptr;
        float64 r = sqrt(x.real * x.real + x.imag * x.imag);
        float64 theta = atan(x.imag/x.real);
        float64 A = y.real * log(r) - y.imag * theta;
        float64 B = y.imag * log(r) + y.real * theta;
        float64 e = exp(A);
        complex128 z = {cos(B), sin(B)};
        (*(complex128 *)z_ptr).real = e * z.real;
        (*(complex128 *)z_ptr).imag = e * z.imag;
        break;
      }
      case ERROR: {
        PyErr_SetString(PyExc_RuntimeError, "something went wrong");
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
      case FP16: {
        float16_t hx = *(float16_t *)x_ptr;
        float16_t hy = *(float16_t *)y_ptr;
        float32 fx = fp16_to_float(hx);
        float32 fy = fp16_to_float(hy);
        float32 fz = powf(fx, fy);
        *(float16_t *)z_ptr = float_to_fp16(fz);
        break;
      }
      case FP32: *(float32 *)z_ptr = (float32)fmodf(*(float32 *)x_ptr, *(float32 *)y_ptr); break;
      case FP64: *(float64 *)z_ptr = (float64)fmodl(*(float64 *)x_ptr, *(float64 *)y_ptr); break;
      case ERROR: {
        PyErr_SetString(PyExc_RuntimeError, "something went wrong");
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
      case FP16: {
        float16_t hx = *(float16_t *)x_ptr;
        float16_t hy = *(float16_t *)y_ptr;
        *(bool *)z_ptr = (hx.bits == hy.bits);
        break;
      }
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
        PyErr_SetString(PyExc_RuntimeError, "something went wrong");
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
      case FP16: {
        float16_t hx = *(float16_t *)x_ptr;
        float16_t hy = *(float16_t *)y_ptr;
        *(bool *)z_ptr = (hx.bits != hy.bits);
        break;
      }
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
      case FP16: {
        float16_t hx = *(float16_t *)x_ptr;
        float16_t hy = *(float16_t *)y_ptr;
        *(bool *)z_ptr = (hx.bits > hy.bits);
        break;
      }
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
      case FP16: {
        float16_t hx = *(float16_t *)x_ptr;
        float16_t hy = *(float16_t *)y_ptr;
        *(bool *)z_ptr = (hx.bits >= hy.bits);
        break;
      }
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
      case FP16: {
        float16_t hx = *(float16_t *)x_ptr;
        float16_t hy = *(float16_t *)y_ptr;
        *(bool *)z_ptr = (hx.bits >= hy.bits);
        break;
      }
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
      case FP16: {
        float16_t hx = *(float16_t *)x_ptr;
        float16_t hy = *(float16_t *)y_ptr;
        *(bool *)z_ptr = (hx.bits >= hy.bits);
        break;
      }
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
      case FP16: {
        float16_t hx = *(float16_t *)x_ptr;
        float32 fx = fp16_to_float(hx);
        *(float16_t *)z_ptr = float_to_fp16(-fx);
        break;
      }
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
      case FP16: {
        float16_t hx = *(float16_t *)x_ptr;
        float32 fx = fp16_to_float(hx);
        *(float16_t *)z_ptr = float_to_fp16(fx);
        break;
      }
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
      case FP16: {
        float16_t hx = *(float16_t *)x_ptr;
        hx.bits &= 0x7FFF;
        *(float16_t *)z_ptr = hx;
        break;
      }
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
        PyErr_SetString(PyExc_RuntimeError, "something went wrong");
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
        PyErr_SetString(PyExc_RuntimeError, "something went wrong");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

// bitwise and kernel
static PyObject *__bitwise_and_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
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
      case INT8: *(int8 *)z_ptr = *(int8 *)x_ptr & *(int8 *)y_ptr; break;
      case UINT8: *(uint8 *)z_ptr = *(uint8 *)x_ptr & *(uint8 *)y_ptr; break;
      case INT16: *(int16 *)z_ptr = *(int16 *)x_ptr & *(int16 *)y_ptr; break;
      case UINT16: *(uint16 *)z_ptr = *(uint16 *)x_ptr & *(uint16 *)y_ptr; break;
      case INT32: *(int32 *)z_ptr = *(int32 *)x_ptr & *(int32 *)y_ptr; break;
      case UINT32: *(uint32 *)z_ptr = *(uint32 *)x_ptr & *(uint32 *)y_ptr; break;
      case INT64: *(int64 *)z_ptr = *(int64 *)x_ptr & *(int64 *)y_ptr; break;
      case UINT64: *(uint64 *)z_ptr = *(uint64 *)x_ptr & *(uint64 *)y_ptr; break;
      case ERROR: {
        PyErr_SetString(PyExc_RuntimeError, "something went wrong");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

// logical and kernel
static PyObject *__logical_and_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  char *px = (char *)tx->buf;
  char *py = (char *)ty->buf;
  char *pz = (char *)tz->buf;
  size_t elem_size = tx->element_size;
  for(size_t i = 0; i < length; i++){
    char *x_ptr = px + i * elem_size;
    char *y_ptr = py + i * elem_size;
    uint8 *z_ptr = (uint8 *)(pz + i);
    switch(dtype){
      case BOOL: *z_ptr = (uint8)((*(bool *)x_ptr != 0) & (*(bool *)y_ptr != 0)); break;
      case INT8: *z_ptr = (uint8)((*(int8 *)x_ptr != 0) & (*(int8 *)y_ptr != 0)); break;
      case UINT8: *z_ptr = (uint8)((*(uint8 *)x_ptr != 0) & (*(uint8 *)y_ptr != 0)); break;
      case INT16: *z_ptr = (uint8)((*(int16 *)x_ptr != 0) & (*(int16 *)y_ptr != 0)); break;
      case UINT16: *z_ptr = (uint8)((*(uint16 *)x_ptr != 0) & (*(uint16 *)y_ptr != 0)); break;
      case INT32: *z_ptr = (uint8)((*(int32 *)x_ptr != 0) & (*(int32 *)y_ptr != 0)); break;
      case UINT32: *z_ptr = (uint8)((*(uint32 *)x_ptr != 0) & (*(uint32 *)y_ptr != 0)); break;
      case INT64: *z_ptr = (uint8)((*(int64 *)x_ptr != 0) & (*(int64 *)y_ptr != 0)); break;
      case UINT64: *z_ptr = (uint8)((*(uint64 *)x_ptr != 0) & (*(uint64 *)y_ptr != 0)); break;
      case FP16: {
        float32 x = fp16_to_float(*(float16 *)x_ptr);
        float32 y = fp16_to_float(*(float16 *)y_ptr);
        *z_ptr = (uint8)((x != 0.0f) & (y != 0.0f));
        break;
      }
      case FP32: *z_ptr = (uint8)((*(float32 *)x_ptr != 0.0f) & (*(float32 *)y_ptr != 0.0f)); break;
      case FP64: *z_ptr = (uint8)((*(float64 *)x_ptr != 0.0) & (*(float64 *)y_ptr != 0.0)); break;
      case CMPX64: {
        complex64 *x = (complex64 *)x_ptr;
        complex64 *y = (complex64 *)y_ptr;
        int x_true = (x->real != 0.0f) || (x->imag != 0.0f);
        int y_true = (y->real != 0.0f) || (y->imag != 0.0f);
        *z_ptr = (uint8)(x_true & y_true);
        break;
      }
      case CMPX128: {
        complex128 *x = (complex128 *)x_ptr;
        complex128 *y = (complex128 *)y_ptr;
        int x_true = (x->real != 0.0) || (x->imag != 0.0);
        int y_true = (y->real != 0.0) || (y->imag != 0.0);
        *z_ptr = (uint8)(x_true & y_true);
        break;
      }
      case ERROR: PyErr_SetString(PyExc_RuntimeError, "Invalid dtype in logical_and"); return NULL;
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

// bitwise nand kernel
static PyObject *__bitwise_nand_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
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
      case BOOL: *(bool *)z_ptr = ~(*(bool *)x_ptr & *(bool *)y_ptr); break;
      case INT8: *(int8 *)z_ptr = ~(*(int8 *)x_ptr & *(int8 *)y_ptr); break;
      case UINT8: *(uint8 *)z_ptr = ~(*(uint8 *)x_ptr & *(uint8 *)y_ptr); break;
      case INT16: *(int16 *)z_ptr = ~(*(int16 *)x_ptr & *(int16 *)y_ptr); break;
      case UINT16: *(uint16 *)z_ptr = ~(*(uint16 *)x_ptr & *(uint16 *)y_ptr); break;
      case INT32: *(int32 *)z_ptr = ~(*(int32 *)x_ptr & *(int32 *)y_ptr); break;
      case UINT32: *(uint32 *)z_ptr = ~(*(uint32 *)x_ptr & *(uint32 *)y_ptr); break;
      case INT64: *(int64 *)z_ptr = ~(*(int64 *)x_ptr & *(int64 *)y_ptr); break;
      case UINT64: *(uint64 *)z_ptr = ~(*(uint64 *)x_ptr & *(uint64 *)y_ptr); break;
      case ERROR: {
        PyErr_SetString(PyExc_RuntimeError, "something went wrong");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

// logical nand kernel
static PyObject *__logical_nand_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  char *px = (char *)tx->buf;
  char *py = (char *)ty->buf;
  char *pz = (char *)tz->buf;
  size_t elem_size = tx->element_size;
  for(size_t i = 0; i < length; i++){
    char *x_ptr = px + i * elem_size;
    char *y_ptr = py + i * elem_size;
    uint8 *z_ptr = (uint8 *)(pz + i);
    switch(dtype){
      case BOOL: *z_ptr = !((*(bool *)x_ptr != 0) & (*(bool *)y_ptr != 0)); break;
      case INT8: *z_ptr = !((*(int8 *)x_ptr != 0) & (*(int8 *)y_ptr != 0)); break;
      case UINT8: *z_ptr = !((*(uint8 *)x_ptr != 0) & (*(uint8 *)y_ptr != 0)); break;
      case INT16: *z_ptr = !((*(int16 *)x_ptr != 0) & (*(int16 *)y_ptr != 0)); break;
      case UINT16: *z_ptr = !((*(uint16 *)x_ptr != 0) & (*(uint16 *)y_ptr != 0)); break;
      case INT32: *z_ptr = !((*(int32 *)x_ptr != 0) & (*(int32 *)y_ptr != 0)); break;
      case UINT32: *z_ptr = !((*(uint32 *)x_ptr != 0) & (*(uint32 *)y_ptr != 0)); break;
      case INT64: *z_ptr = !((*(int64 *)x_ptr != 0) & (*(int64 *)y_ptr != 0)); break;
      case UINT64: *z_ptr = !((*(uint64 *)x_ptr != 0) & (*(uint64 *)y_ptr != 0)); break;
      case FP16: {
        float32 x = fp16_to_float(*(float16 *)x_ptr);
        float32 y = fp16_to_float(*(float16 *)y_ptr);
        *z_ptr = !((x != 0.0f) & (y != 0.0f));
        break;
      }
      case FP32: *z_ptr = !((*(float32 *)x_ptr != 0.0f) & (*(float32 *)y_ptr != 0.0f)); break;
      case FP64: *z_ptr = !((*(float64 *)x_ptr != 0.0) & (*(float64 *)y_ptr != 0.0)); break;
      case CMPX64: {
        complex64 *x = (complex64 *)x_ptr;
        complex64 *y = (complex64 *)y_ptr;
        int x_true = (x->real != 0.0f) || (x->imag != 0.0f);
        int y_true = (y->real != 0.0f) || (y->imag != 0.0f);
        *z_ptr = !(x_true & y_true);
        break;
      }
      case CMPX128: {
        complex128 *x = (complex128 *)x_ptr;
        complex128 *y = (complex128 *)y_ptr;
        int x_true = (x->real != 0.0) || (x->imag != 0.0);
        int y_true = (y->real != 0.0) || (y->imag != 0.0);
        *z_ptr = !(x_true & y_true);
        break;
      }
      case ERROR: PyErr_SetString(PyExc_RuntimeError, "Invalid dtype in logical_and"); return NULL;
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

// bitwise or kernel
static PyObject *__bitwise_or_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
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
      case BOOL: *(bool *)z_ptr = *(bool *)x_ptr | *(bool *)y_ptr; break;
      case INT8: *(int8 *)z_ptr = *(int8 *)x_ptr | *(int8 *)y_ptr; break;
      case UINT8: *(uint8 *)z_ptr = *(uint8 *)x_ptr | *(uint8 *)y_ptr; break;
      case INT16: *(int16 *)z_ptr = *(int16 *)x_ptr | *(int16 *)y_ptr; break;
      case UINT16: *(uint16 *)z_ptr = *(uint16 *)x_ptr | *(uint16 *)y_ptr; break;
      case INT32: *(int32 *)z_ptr = *(int32 *)x_ptr | *(int32 *)y_ptr; break;
      case UINT32: *(uint32 *)z_ptr = *(uint32 *)x_ptr | *(uint32 *)y_ptr; break;
      case INT64: *(int64 *)z_ptr = *(int64 *)x_ptr | *(int64 *)y_ptr; break;
      case UINT64: *(uint64 *)z_ptr = *(uint64 *)x_ptr | *(uint64 *)y_ptr; break;
      case ERROR: {
        PyErr_SetString(PyExc_RuntimeError, "something went wrong");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

// logical or kernel
static PyObject *__logical_or_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  char *px = (char *)tx->buf;
  char *py = (char *)ty->buf;
  char *pz = (char *)tz->buf;
  size_t elem_size = tx->element_size;
  for(size_t i = 0; i < length; i++){
    char *x_ptr = px + i * elem_size;
    char *y_ptr = py + i * elem_size;
    uint8 *z_ptr = (uint8 *)(pz + i);
    switch(dtype){
      case BOOL: *z_ptr = ((*(bool *)x_ptr != 0) | (*(bool *)y_ptr != 0)); break;
      case INT8: *z_ptr = ((*(int8 *)x_ptr != 0) | (*(int8 *)y_ptr != 0)); break;
      case UINT8: *z_ptr = ((*(uint8 *)x_ptr != 0) | (*(uint8 *)y_ptr != 0)); break;
      case INT16: *z_ptr = ((*(int16 *)x_ptr != 0) | (*(int16 *)y_ptr != 0)); break;
      case UINT16: *z_ptr = ((*(uint16 *)x_ptr != 0) | (*(uint16 *)y_ptr != 0)); break;
      case INT32: *z_ptr = ((*(int32 *)x_ptr != 0) | (*(int32 *)y_ptr != 0)); break;
      case UINT32: *z_ptr = ((*(uint32 *)x_ptr != 0) | (*(uint32 *)y_ptr != 0)); break;
      case INT64: *z_ptr = ((*(int64 *)x_ptr != 0) | (*(int64 *)y_ptr != 0)); break;
      case UINT64: *z_ptr = ((*(uint64 *)x_ptr != 0) | (*(uint64 *)y_ptr != 0)); break;
      case FP16: {
        float32 x = fp16_to_float(*(float16 *)x_ptr);
        float32 y = fp16_to_float(*(float16 *)y_ptr);
        *z_ptr = ((x != 0.0f) | (y != 0.0f));
        break;
      }
      case FP32: *z_ptr = ((*(float32 *)x_ptr != 0.0f) | (*(float32 *)y_ptr != 0.0f)); break;
      case FP64: *z_ptr = ((*(float64 *)x_ptr != 0.0) | (*(float64 *)y_ptr != 0.0)); break;
      case CMPX64: {
        complex64 *x = (complex64 *)x_ptr;
        complex64 *y = (complex64 *)y_ptr;
        int x_true = (x->real != 0.0f) || (x->imag != 0.0f);
        int y_true = (y->real != 0.0f) || (y->imag != 0.0f);
        *z_ptr = (x_true | y_true);
        break;
      }
      case CMPX128: {
        complex128 *x = (complex128 *)x_ptr;
        complex128 *y = (complex128 *)y_ptr;
        int x_true = (x->real != 0.0) || (x->imag != 0.0);
        int y_true = (y->real != 0.0) || (y->imag != 0.0);
        *z_ptr = (x_true | y_true);
        break;
      }
      case ERROR: PyErr_SetString(PyExc_RuntimeError, "Invalid dtype in logical_and"); return NULL;
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

// bitwise nor kernel
static PyObject *__bitwise_nor_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
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
      case BOOL: *(bool *)z_ptr = ~(*(bool *)x_ptr | *(bool *)y_ptr); break;
      case INT8: *(int8 *)z_ptr = ~(*(int8 *)x_ptr | *(int8 *)y_ptr); break;
      case UINT8: *(uint8 *)z_ptr = ~(*(uint8 *)x_ptr | *(uint8 *)y_ptr); break;
      case INT16: *(int16 *)z_ptr = ~(*(int16 *)x_ptr | *(int16 *)y_ptr); break;
      case UINT16: *(uint16 *)z_ptr = ~(*(uint16 *)x_ptr | *(uint16 *)y_ptr); break;
      case INT32: *(int32 *)z_ptr = ~(*(int32 *)x_ptr | *(int32 *)y_ptr); break;
      case UINT32: *(uint32 *)z_ptr = ~(*(uint32 *)x_ptr | *(uint32 *)y_ptr); break;
      case INT64: *(int64 *)z_ptr = ~(*(int64 *)x_ptr | *(int64 *)y_ptr); break;
      case UINT64: *(uint64 *)z_ptr = ~(*(uint64 *)x_ptr | *(uint64 *)y_ptr); break;
      case ERROR: {
        PyErr_SetString(PyExc_RuntimeError, "something went wrong");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

// logical nor kernel
static PyObject *__logical_nor_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  char *px = (char *)tx->buf;
  char *py = (char *)ty->buf;
  char *pz = (char *)tz->buf;
  size_t elem_size = tx->element_size;
  for(size_t i = 0; i < length; i++){
    char *x_ptr = px + i * elem_size;
    char *y_ptr = py + i * elem_size;
    uint8 *z_ptr = (uint8 *)(pz + i);
    switch(dtype){
      case INT8: *z_ptr = !((*(int8 *)x_ptr != 0) | (*(int8 *)y_ptr != 0)); break;
      case UINT8: *z_ptr = !((*(uint8 *)x_ptr != 0) | (*(uint8 *)y_ptr != 0)); break;
      case INT16: *z_ptr = !((*(int16 *)x_ptr != 0) | (*(int16 *)y_ptr != 0)); break;
      case UINT16: *z_ptr = !((*(uint16 *)x_ptr != 0) | (*(uint16 *)y_ptr != 0)); break;
      case INT32: *z_ptr = !((*(int32 *)x_ptr != 0) | (*(int32 *)y_ptr != 0)); break;
      case UINT32: *z_ptr = !((*(uint32 *)x_ptr != 0) | (*(uint32 *)y_ptr != 0)); break;
      case INT64: *z_ptr = !((*(int64 *)x_ptr != 0) | (*(int64 *)y_ptr != 0)); break;
      case UINT64: *z_ptr = !((*(uint64 *)x_ptr != 0) | (*(uint64 *)y_ptr != 0)); break;
      case FP16: {
        float32 x = fp16_to_float(*(float16 *)x_ptr);
        float32 y = fp16_to_float(*(float16 *)y_ptr);
        *z_ptr = !((x != 0.0f) | (y != 0.0f));
        break;
      }
      case FP32: *z_ptr = !((*(float32 *)x_ptr != 0.0f) | (*(float32 *)y_ptr != 0.0f)); break;
      case FP64: *z_ptr = !((*(float64 *)x_ptr != 0.0) | (*(float64 *)y_ptr != 0.0)); break;
      case CMPX64: {
        complex64 *x = (complex64 *)x_ptr;
        complex64 *y = (complex64 *)y_ptr;
        int x_true = (x->real != 0.0f) || (x->imag != 0.0f);
        int y_true = (y->real != 0.0f) || (y->imag != 0.0f);
        *z_ptr = !(x_true | y_true);
        break;
      }
      case CMPX128: {
        complex128 *x = (complex128 *)x_ptr;
        complex128 *y = (complex128 *)y_ptr;
        int x_true = (x->real != 0.0) || (x->imag != 0.0);
        int y_true = (y->real != 0.0) || (y->imag != 0.0);
        *z_ptr = !(x_true | y_true);
        break;
      }
      case ERROR: PyErr_SetString(PyExc_RuntimeError, "Invalid dtype in logical_and"); return NULL;
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

// bitwise not kernel
static PyObject *__bitwise_not_tensor__(const tensor_t *tx, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  char *px = (char *)tx->buf;
  char *pz = (char *)tz->buf;
  size_t elem_size = tx->element_size;
  for(size_t i = 0; i < length; i++){
    char *x_ptr = px + i * elem_size;
    char *z_ptr = pz + i * elem_size;
    switch(dtype){
      case BOOL: *(bool *)z_ptr = ~(*(bool *)x_ptr); break;
      case INT8: *(int8 *)z_ptr = ~(*(int8 *)x_ptr); break;
      case UINT8: *(uint8 *)z_ptr = ~(*(uint8 *)x_ptr); break;
      case INT16: *(int16 *)z_ptr = ~(*(int16 *)x_ptr); break;
      case UINT16: *(uint16 *)z_ptr = ~(*(uint16 *)x_ptr); break;
      case INT32: *(int32 *)z_ptr = ~(*(int32 *)x_ptr); break;
      case UINT32: *(uint32 *)z_ptr = ~(*(uint32 *)x_ptr); break;
      case INT64: *(int64 *)z_ptr = ~(*(int64 *)x_ptr); break;
      case UINT64: *(uint64 *)z_ptr = ~(*(uint64 *)x_ptr); break;
      case ERROR: {
        PyErr_SetString(PyExc_RuntimeError, "something went wrong");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

// logical not kernel
static PyObject *__logical_not_tensor__(const tensor_t *tx, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  char *px = (char *)tx->buf;
  char *pz = (char *)tz->buf;
  size_t elem_size = tx->element_size;
  for(size_t i = 0; i < length; i++){
    char *x_ptr = px + i * elem_size;
    uint8 *z_ptr = (uint8 *)(pz + i);
    uint8 xt = 0;
    switch(dtype){
      case BOOL: xt = (*(bool *)x_ptr != 0); break;
      case INT8: xt = (*(int8 *)x_ptr != 0); break;
      case UINT8: xt = (*(uint8 *)x_ptr != 0); break;
      case INT16: xt = (*(int16 *)x_ptr != 0); break;
      case UINT16: xt = (*(uint16 *)x_ptr != 0); break;
      case INT32: xt = (*(int32 *)x_ptr != 0); break;
      case UINT32: xt = (*(uint32 *)x_ptr != 0); break;
      case INT64: xt = (*(int64 *)x_ptr != 0); break;
      case UINT64: xt = (*(uint64 *)x_ptr != 0); break;
      case FP16: {
        float32 v = fp16_to_float(*(float16 *)x_ptr);
        xt = (v != 0.0f);
        break;
      }
      case FP32: xt = (*(float32 *)x_ptr != 0.0f); break;
      case FP64: xt = (*(float64 *)x_ptr != 0.0); break;
      case CMPX64: {
        complex64 *x = (complex64 *)x_ptr;
        xt = (x->real != 0.0f || x->imag != 0.0f);
        break;
      }
      case CMPX128: {
        complex128 *x = (complex128 *)x_ptr;
        xt = (x->real != 0.0 || x->imag != 0.0);
        break;
      }
      default:
        PyErr_SetString(PyExc_RuntimeError, "Unsupported dtype in logical_not");
        return NULL;
    }
    *z_ptr = !xt;
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

// bitwise xor kernel
static PyObject *__bitwise_xor_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
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
      case BOOL: *(bool *)z_ptr = *(bool *)x_ptr ^ *(bool *)y_ptr; break;
      case INT8: *(int8 *)z_ptr = *(int8 *)x_ptr ^ *(int8 *)y_ptr; break;
      case UINT8: *(uint8 *)z_ptr = *(uint8 *)x_ptr ^ *(uint8 *)y_ptr; break;
      case INT16: *(int16 *)z_ptr = *(int16 *)x_ptr ^ *(int16 *)y_ptr; break;
      case UINT16: *(uint16 *)z_ptr = *(uint16 *)x_ptr ^ *(uint16 *)y_ptr; break;
      case INT32: *(int32 *)z_ptr = *(int32 *)x_ptr ^ *(int32 *)y_ptr; break;
      case UINT32: *(uint32 *)z_ptr = *(uint32 *)x_ptr ^ *(uint32 *)y_ptr; break;
      case INT64: *(int64 *)z_ptr = *(int64 *)x_ptr ^ *(int64 *)y_ptr; break;
      case UINT64: *(uint64 *)z_ptr = *(uint64 *)x_ptr ^ *(uint64 *)y_ptr; break;
      case ERROR: {
        PyErr_SetString(PyExc_RuntimeError, "something went wrong");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

// logical xor kernel
static PyObject *__logical_xor_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  char *px = (char *)tx->buf;
  char *py = (char *)ty->buf;
  char *pz = (char *)tz->buf;
  size_t elem_size = tx->element_size;
  for(size_t i = 0; i < length; i++){
    char *x_ptr = px + i * elem_size;
    char *y_ptr = py + i * elem_size;
    uint8 *z_ptr = (uint8 *)(pz + i);
    uint8 xt = 0, yt = 0;
    switch(dtype){
      case BOOL: xt = (*(bool *)x_ptr != 0); yt = (*(bool *)y_ptr != 0); break;
      case INT8: xt = (*(int8 *)x_ptr != 0); yt = (*(int8 *)y_ptr != 0); break;
      case UINT8: xt = (*(uint8 *)x_ptr != 0); yt = (*(uint8 *)y_ptr != 0); break;
      case INT16: xt = (*(int16 *)x_ptr != 0); yt = (*(int16 *)y_ptr != 0); break;
      case UINT16: xt = (*(uint16 *)x_ptr != 0); yt = (*(uint16 *)y_ptr != 0); break;
      case INT32: xt = (*(int32 *)x_ptr != 0); yt = (*(int32 *)y_ptr != 0); break;
      case UINT32: xt = (*(uint32 *)x_ptr != 0); yt = (*(uint32 *)y_ptr != 0); break;
      case INT64: xt = (*(int64 *)x_ptr != 0); yt = (*(int64 *)y_ptr != 0); break;
      case UINT64: xt = (*(uint64 *)x_ptr != 0); yt = (*(uint64 *)y_ptr != 0); break;
      case FP16: {
        float32 xv = fp16_to_float(*(float16 *)x_ptr);
        float32 yv = fp16_to_float(*(float16 *)y_ptr);
        xt = (xv != 0.0f);
        yt = (yv != 0.0f);
        break;
      }
      case FP32:
        xt = (*(float32 *)x_ptr != 0.0f);
        yt = (*(float32 *)y_ptr != 0.0f);
        break;
      case FP64:
        xt = (*(float64 *)x_ptr != 0.0);
        yt = (*(float64 *)y_ptr != 0.0);
        break;
      case CMPX64: {
        complex64 *x = (complex64 *)x_ptr;
        complex64 *y = (complex64 *)y_ptr;
        xt = (x->real != 0.0f || x->imag != 0.0f);
        yt = (y->real != 0.0f || y->imag != 0.0f);
        break;
      }
      case CMPX128: {
        complex128 *x = (complex128 *)x_ptr;
        complex128 *y = (complex128 *)y_ptr;
        xt = (x->real != 0.0 || x->imag != 0.0);
        yt = (y->real != 0.0 || y->imag != 0.0);
        break;
      }
      default:
        PyErr_SetString(PyExc_RuntimeError, "Unsupported dtype in logical_xor");
        return NULL;
    }
    *z_ptr = xt ^ yt;
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

// bitwise xnor kernel
static PyObject *__bitwise_xnor_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
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
      case BOOL: *(bool *)z_ptr = ~(*(bool *)x_ptr ^ *(bool *)y_ptr); break;
      case INT8: *(int8 *)z_ptr = ~(*(int8 *)x_ptr ^ *(int8 *)y_ptr); break;
      case UINT8: *(uint8 *)z_ptr = ~(*(uint8 *)x_ptr ^ *(uint8 *)y_ptr); break;
      case INT16: *(int16 *)z_ptr = ~(*(int16 *)x_ptr ^ *(int16 *)y_ptr); break;
      case UINT16: *(uint16 *)z_ptr = ~(*(uint16 *)x_ptr ^ *(uint16 *)y_ptr); break;
      case INT32: *(int32 *)z_ptr = ~(*(int32 *)x_ptr ^ *(int32 *)y_ptr); break;
      case UINT32: *(uint32 *)z_ptr = ~(*(uint32 *)x_ptr ^ *(uint32 *)y_ptr); break;
      case INT64: *(int64 *)z_ptr = ~(*(int64 *)x_ptr ^ *(int64 *)y_ptr); break;
      case UINT64: *(uint64 *)z_ptr = ~(*(uint64 *)x_ptr ^ *(uint64 *)y_ptr); break;
      case ERROR: {
        PyErr_SetString(PyExc_RuntimeError, "something went wrong");
        return NULL;
      }
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

//logical xnor kernel
static PyObject *__logical_xnor_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->size;
  dtype_t dtype = tx->dtype;
  char *px = (char *)tx->buf;
  char *py = (char *)ty->buf;
  char *pz = (char *)tz->buf;
  size_t elem_size = tx->element_size;
  for(size_t i = 0; i < length; i++){
    char *x_ptr = px + i * elem_size;
    char *y_ptr = py + i * elem_size;
    uint8 *z_ptr = (uint8 *)(pz + i);
    uint8 xt = 0, yt = 0;
    switch(dtype){
      case BOOL: xt = (*(bool *)x_ptr != 0); yt = (*(bool *)y_ptr != 0); break;
      case INT8: xt = (*(int8 *)x_ptr != 0); yt = (*(int8 *)y_ptr != 0); break;
      case UINT8: xt = (*(uint8 *)x_ptr != 0); yt = (*(uint8 *)y_ptr != 0); break;
      case INT16: xt = (*(int16 *)x_ptr != 0); yt = (*(int16 *)y_ptr != 0); break;
      case UINT16: xt = (*(uint16 *)x_ptr != 0); yt = (*(uint16 *)y_ptr != 0); break;
      case INT32: xt = (*(int32 *)x_ptr != 0); yt = (*(int32 *)y_ptr != 0); break;
      case UINT32: xt = (*(uint32 *)x_ptr != 0); yt = (*(uint32 *)y_ptr != 0); break;
      case INT64: xt = (*(int64 *)x_ptr != 0); yt = (*(int64 *)y_ptr != 0); break;
      case UINT64: xt = (*(uint64 *)x_ptr != 0); yt = (*(uint64 *)y_ptr != 0); break;
      case FP16: {
        float32 xv = fp16_to_float(*(float16 *)x_ptr);
        float32 yv = fp16_to_float(*(float16 *)y_ptr);
        xt = (xv != 0.0f);
        yt = (yv != 0.0f);
        break;
      }
      case FP32:
        xt = (*(float32 *)x_ptr != 0.0f);
        yt = (*(float32 *)y_ptr != 0.0f);
        break;
      case FP64:
        xt = (*(float64 *)x_ptr != 0.0);
        yt = (*(float64 *)y_ptr != 0.0);
        break;
      case CMPX64: {
        complex64 *x = (complex64 *)x_ptr;
        complex64 *y = (complex64 *)y_ptr;
        xt = (x->real != 0.0f || x->imag != 0.0f);
        yt = (y->real != 0.0f || y->imag != 0.0f);
        break;
      }
      case CMPX128: {
        complex128 *x = (complex128 *)x_ptr;
        complex128 *y = (complex128 *)y_ptr;
        xt = (x->real != 0.0 || x->imag != 0.0);
        yt = (y->real != 0.0 || y->imag != 0.0);
        break;
      }
      default:
        PyErr_SetString(PyExc_RuntimeError, "Unsupported dtype in logical_xnor");
        return NULL;
    }
    *z_ptr = !(xt ^ yt);
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
    case FP16: {
      float32 acc = 0.0f;
      for(size_t i = 0; i < length; i++){
        char *x_ptr = px + i * elem_size;
        float16_t hx = *(float16_t *)x_ptr;
        acc += fp16_to_float(hx);
      }
      *(float16_t *)tz->buf = float_to_fp16(acc);
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
      PyErr_SetString(PyExc_RuntimeError, "something went wrong");
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

#define SUM_AXIS_FP16_KERNEL(NAME, ACC_T)                                           \
static PyObject *__sum_axis_##NAME##__(const tensor_t *tx, tensor_t *tz, int axis){ \
  float16_t *in  = (float16_t*)tx->buf;                                             \
  float16_t *out = (float16_t*)tz->buf;                                             \
  int reduced_dim = tx->shape[axis];                                                \
  for(int out_index = 0; out_index < tz->size; out_index++){                        \
    int tmp = out_index;                                                            \
    int idx_in[16] = {0};                                                           \
    int j = 0;                                                                      \
    for(int i = 0; i < tx->ndim; i++){                                              \
      if(i == axis) continue;                                                       \
      idx_in[i] = tmp / tz->stride[j];                                              \
      tmp %= tz->stride[j];                                                         \
      j++;                                                                          \
    }                                                                               \
    ACC_T total = 0;                                                                \
    for(int r = 0; r < reduced_dim; r++){                                           \
      idx_in[axis] = r;                                                             \
      int in_offset = 0;                                                            \
      for(int k = 0; k < tx->ndim; k++){                                            \
        in_offset += idx_in[k] * tx->stride[k];                                     \
      }                                                                             \
      total += (ACC_T)fp16_to_float(in[in_offset]);                                 \
    }                                                                               \
    out[out_index] = float_to_fp16((float)total);                                   \
  }                                                                                 \
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);                   \
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

static dtype_t promote_div(dtype_t dt) {
  switch (dt) {
    case INT8:
    case UINT8:
    case INT16:
    case UINT16:
      return FP16;
    case INT32:
    case UINT32:
      return FP32;
    case INT64:
    case UINT64:
      return FP64;
    case FP16:
    case FP32:
    case FP64:
      return dt;
    default:
      return ERROR;
  }
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
    tensor_t *tz = tt_malloc(sizeof(tensor_t));
    if(!tz){
      PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
      return NULL;
    }
    tz->dtype = (tx->dtype == CMPX64) ? CMPX64 : CMPX128;
    tz->size = tx->size;
    tz->ndim = tx->ndim;
    tz->element_size = getsize(tz->dtype);
    tz->device = (device_t){CPU, 0};
    if(tx->ndim > 0 && tx->shape){
      tz->shape = tt_malloc(sizeof(size_t) * tz->ndim);
      tz->stride = tt_malloc(sizeof(size_t) * tz->ndim);
      if(!tz->shape || !tz->stride){
        destroy(tz);
        PyErr_SetString(PyExc_RuntimeError, "tensor_t.shape or tensor_t.stride allocation failed!");
        return NULL;
      }
      for(size_t i = 0; i < tz->ndim; i++){
        tz->shape[i] = tx->shape[i];
        tz->stride[i] = tx->stride[i];
      }
    }else{
      tz->shape = NULL;
      tz->stride = NULL;
    }
    tz->storage = tt_malloc(sizeof(storage_t));
    if(!tz->storage){
      destroy(tz);
      PyErr_SetString(PyExc_RuntimeError, "tensor_t.storage allocation failed!");
      return NULL;
    }
    tz->storage->bytes = tz->size * tz->element_size;
    tz->storage->device = tz->device;
    tz->storage->refcount = 1;
    tz->storage->ptr = tt_malloc(tz->storage->bytes);
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
    tensor_t *tz = tt_malloc(sizeof(tensor_t));
    if(!tz){
      PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
      return NULL;
    }
    tz->dtype = promote_div(tx->dtype);
    tz->size = tx->size;
    tz->ndim = tx->ndim;
    tz->element_size = getsize(tz->dtype);
    tz->device = (device_t){CPU, 0};
    if(tx->ndim > 0 && tx->shape){
      tz->shape = tt_malloc(sizeof(size_t) * tz->ndim);
      tz->stride = tt_malloc(sizeof(size_t) * tz->ndim);
      if(!tz->shape || !tz->stride){
        free(tz);
        PyErr_SetString(PyExc_RuntimeError, "tensor_t.shape or tensor_t.stride allocation failed!");
        return NULL;
      }
      for(size_t i = 0; i < tz->ndim; i++){
        tz->shape[i] = tx->shape[i];
        tz->stride[i] = tx->stride[i];
      }
    }else{
      tz->shape = NULL;
      tz->stride = NULL;
    }
    tz->storage = tt_malloc(sizeof(storage_t));
    if(!tz->storage){
      if(tz->shape) free(tz->shape);
      free(tz);
      PyErr_SetString(PyExc_RuntimeError, "tensor_t.storage allocation failed!");
      return NULL;
    }
    tz->storage->bytes = tz->size * tz->element_size;
    tz->storage->device = tz->device;
    tz->storage->refcount = 1;
    tz->storage->ptr = tt_malloc(tz->storage->bytes);
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
    PyErr_SetString(PyExc_TypeError, "fdiv() is not supported on complex tensor(s)");
    return NULL;
  }
  tensor_t *tz = tt_malloc(sizeof(tensor_t));
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
    return NULL;
  }
  tz->dtype = INT64; // assuming fp32 and fp64 will get the work done
  tz->size = tx->size;
  tz->ndim = tx->ndim;
  tz->element_size = sizeof(int64);
  tz->device = (device_t){CPU, 0};
  if(tx->ndim > 0 && tx->shape){
    tz->shape = tt_malloc(sizeof(size_t) * tz->ndim);
    tz->stride = tt_malloc(sizeof(size_t) * tz->ndim);
    if(!tz->shape || !tz->stride){
      destroy(tz);
      PyErr_SetString(PyExc_RuntimeError, "tensor_t.shape or tensor_t.stride allocation failed!");
      return NULL;
    }
    for(size_t i = 0; i < tz->ndim; i++){
      tz->shape[i] = tx->shape[i];
      tz->stride[i] = tx->stride[i];
    }
  }else{
    tz->shape = NULL;
    tz->stride = NULL;
  }
  tz->storage = tt_malloc(sizeof(storage_t));
  if(!tz->storage){
    if(tz->shape) free(tz->shape);
    free(tz);
    PyErr_SetString(PyExc_RuntimeError, "tensor_t.storage allocation failed!");
    return NULL;
  }
  tz->storage->bytes = tz->size * tz->element_size;
  tz->storage->device = tz->device;
  tz->storage->refcount = 1;
  tz->storage->ptr = tt_malloc(tz->storage->bytes);
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

static PyObject *__sqrt_tensor__(const tensor_t *tx, tensor_t *tz){
  size_t N = tx->size;
  switch(tz->dtype){
    case FP16: {
      float16 *in = (float16 *)tx->buf;
      float16 *out = (float16 *)tz->buf;
      for(size_t i = 0; i < N; i++){ out[i] = float_to_fp16(sqrtf(fp16_to_float(in[i]))); }
      break;
    }
    case FP32: {
      float32 *out = (float32 *)tz->buf;
      switch (tx->dtype){
        case INT8: { int8 *in = (int8 *)tx->buf; for(size_t i = 0; i < N; i++){ out[i] = sqrtf((float32)in[i]); break; }}
        case UINT8: { uint8 *in = (uint8 *)tx->buf; for(size_t i = 0; i < N; i++){ out[i] = sqrtf((float32)in[i]); break; }}
        case INT16: { int16 *in = (int16 *)tx->buf; for(size_t i = 0; i < N; i++){ out[i] = sqrtf((float32)in[i]); break; }}
        case UINT16: { uint16 *in = (uint16 *)tx->buf; for(size_t i = 0; i < N; i++){ out[i] = sqrtf((float32)in[i]); break; }}
        case INT32: { int32 *in = (int32 *)tx->buf; for(size_t i = 0; i < N; i++){ out[i] = sqrtf((float32)in[i]); break; }}
        case UINT32: { uint32 *in = (uint32 *)tx->buf; for(size_t i = 0; i < N; i++){ out[i] = sqrtf((float32)in[i]); break; }}
        case FP32: { float32 *in = (float32 *)tx->buf; for(size_t i = 0; i < N; i++){ out[i] = sqrtf(in[i]); break; }}
        default: PyErr_SetString(PyExc_TypeError, "cos_() unsupported dtype"); return NULL;
      }
      break;
    }
    case FP64: {
      float64 *in  = (float64 *)tx->buf;
      float64 *out = (float64 *)tz->buf;
      for(size_t i = 0; i < N; i++){ out[i] = sqrt(in[i]); }
      break;
    }
    case CMPX64: {
      complex64 *in = (complex64 *)tx->buf;
      complex64 *out = (complex64 *)tz->buf;
      for(size_t i = 0; i < N; i++){
        float32 a = in[i].real;
        float32 b = in[i].imag;
        float32 r = sqrtf(a*a+b*b);
        float32 real_part = sqrtf((r+a) * 0.5f);
        float32 imag_part = sqrtf((r-a) * 0.5f);
        if(b < a){ imag_part = -imag_part; }
        out[i].real = real_part;
        out[i].imag = imag_part;
      }
      break;
    }
    case CMPX128: {
      complex128 *in = (complex128 *)tx->buf;
      complex128 *out = (complex128 *)tz->buf;
      for(size_t i = 0; i < N; i++){
        float64 a = in[i].real;
        float64 b = in[i].imag;
        float64 r = sqrt(a*a+b*b);
        float64 real_part = sqrt((r+a) * 0.5);
        float64 imag_part = sqrt((r-a) * 0.5);
        if(b < a){ imag_part = -imag_part; }
        out[i].real = real_part;
        out[i].imag = imag_part;
      }
      break;
    }
    default:
      PyErr_SetString(PyExc_TypeError, "Unsupported dtype in __sqrt_tensor__");
      return NULL;
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *sqrt_(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "operands must be the tensor capsule");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  dtype_t dtype;
  switch(tx->dtype){
    case INT8:
    case UINT8:
    case INT16:
    case UINT16:
    case INT32:
    case UINT32: dtype = FP32; break;
    case FP16: dtype=FP16; break;
    case FP32: dtype=FP32; break;
    case FP64: dtype=FP64; break;
    case CMPX64: dtype=CMPX64; break;
    case CMPX128: dtype=CMPX128; break;
    default:
      PyErr_SetString(PyExc_TypeError, "Unsupported dtype for sqrt_()");
      return NULL;
  }
  tensor_t *tz = tt_malloc(sizeof(tensor_t));
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
    return NULL;
  }
  tz->dtype = dtype;
  tz->element_size = getsize(dtype);
  tz->device = (device_t){CPU,0};
  tz->ndim = tx->ndim;
  tz->size = tx->size;
  if(tx->ndim > 0 && tx->shape){
    tz->shape = tt_malloc(sizeof(size_t) * tz->ndim);
    tz->stride = tt_malloc(sizeof(size_t) * tz->ndim);
    if(!tz->shape || !tz->stride){
      destroy(tz);
      PyErr_SetString(PyExc_RuntimeError, "tensor_t.shape or tensor_t.stride allocation failed!");
      return NULL;
    }
    for(size_t i = 0; i < tz->ndim; i++){
      tz->shape[i] = tx->shape[i];
      tz->stride[i] = tx->stride[i];
    }
  }else{
    tz->shape = NULL;
    tz->stride = NULL;
  }
  tz->storage = tt_malloc(sizeof(storage_t));
  if(!tz->storage){
    destroy(tz);
    PyErr_SetString(PyExc_RuntimeError, "tensor_t.storage_t allocation failed!");
    return NULL;
  }
  tz->storage->bytes = tz->size * tz->element_size;
  tz->storage->device = tz->device;
  tz->storage->refcount = 1;
  tz->storage->ptr = tt_malloc(tz->storage->bytes);
  if(!tz->storage->ptr){
    destroy(tz);
    PyErr_SetString(PyExc_RuntimeError, "tensor_t.storage_t.ptr allocation failed!");
    return NULL;
  }
  tz->buf = tz->storage->ptr;
  return __sqrt_tensor__(tx, tz);
}

static PyObject *__cbrt_tensor__(const tensor_t *tx, tensor_t *tz){
  size_t N = tx->size;
  switch(tz->dtype){
    case FP16: {
      float16 *in  = (float16*)tx->buf;
      float16 *out = (float16*)tz->buf;
      for(size_t i = 0; i < N; i++){
        out[i] = float_to_fp16(cbrtf(fp16_to_float(in[i])));
      }
      break;
    }
    case FP32: {
      float32 *out = (float32*)tz->buf;
      switch(tx->dtype){
        case INT8: { int8 *in=(int8*)tx->buf; for(size_t i=0;i<N;i++) out[i]=cbrtf((float32)in[i]); break; }
        case UINT8: { uint8 *in=(uint8*)tx->buf; for(size_t i=0;i<N;i++) out[i]=cbrtf((float32)in[i]); break; }
        case INT16: { int16 *in=(int16*)tx->buf; for(size_t i=0;i<N;i++) out[i]=cbrtf((float32)in[i]); break; }
        case UINT16: { uint16*in=(uint16*)tx->buf; for(size_t i=0;i<N;i++) out[i]=cbrtf((float32)in[i]); break; }
        case INT32: { int32 *in=(int32*)tx->buf; for(size_t i=0;i<N;i++) out[i]=cbrtf((float32)in[i]); break; }
        case UINT32: { uint32*in=(uint32*)tx->buf; for(size_t i=0;i<N;i++) out[i]=cbrtf((float32)in[i]); break; }
        case FP32: { float32*in=(float32*)tx->buf;for(size_t i=0;i<N;i++) out[i]=cbrtf(in[i]); break; }
        default:
          PyErr_SetString(PyExc_TypeError,"Unexpected source dtype for FP32 cbrt");
          return NULL;
      }
      break;
    }
    case FP64: {
      float64 *in  = (float64*)tx->buf;
      float64 *out = (float64*)tz->buf;
      for(size_t i = 0; i < N; i++){ out[i] = cbrt(in[i]); }
      break;
    }
    case CMPX64: {
      complex64 *in  = (complex64*)tx->buf;
      complex64 *out = (complex64*)tz->buf;
      for(size_t i = 0; i < N; i++){
        float32 a = in[i].real;
        float32 b = in[i].imag;
        float32 r = hypotf(a,b);
        float32 theta = atan2f(b,a);
        float32 root_r = cbrtf(r);
        float32 angle = theta / 3.0f;
        out[i].real = root_r * cosf(angle);
        out[i].imag = root_r * sinf(angle);
      }
      break;
    }
    case CMPX128: {
      complex128 *in  = (complex128*)tx->buf;
      complex128 *out = (complex128*)tz->buf;
      for(size_t i = 0; i < N; i++){
        float64 a = in[i].real;
        float64 b = in[i].imag;
        float64 r = hypot(a,b);
        float64 theta = atan2(b,a);
        float64 root_r = cbrt(r);
        float64 angle = theta / 3.0;
        out[i].real = root_r * cos(angle);
        out[i].imag = root_r * sin(angle);
      }
      break;
    }
    default:
      PyErr_SetString(PyExc_TypeError,"Unsupported dtype in __cbrt_tensor__");
      return NULL;
  }
  return PyCapsule_New(tz,"tensor_t on CPU",capsule_destroyer);
}

static PyObject *cbrt_(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "operands must be the tensor capsule");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  dtype_t dtype;
  switch(tx->dtype){
    case INT8:
    case UINT8:
    case INT16:
    case UINT16:
    case INT32:
    case UINT32: dtype = FP32; break;
    case FP16: dtype=FP16; break;
    case FP32: dtype=FP32; break;
    case FP64: dtype=FP64; break;
    case CMPX64: dtype=CMPX64; break;
    case CMPX128: dtype=CMPX128; break;
    default: PyErr_SetString(PyExc_TypeError, "unsupported dtype for cbrt_()"); return NULL;
  }
  tensor_t *tz = tt_malloc(sizeof(tensor_t));
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
    return NULL;
  }
  tz->dtype = dtype;
  tz->element_size = getsize(dtype);
  tz->device = (device_t){CPU,0};
  tz->ndim = tx->ndim;
  tz->size = tx->size;
  if(tx->ndim > 0 && tx->shape){
    tz->shape = tt_malloc(sizeof(size_t) * tz->ndim);
    tz->stride = tt_malloc(sizeof(size_t) * tz->ndim);
    if(!tz->shape || !tz->stride){
      destroy(tz);
      PyErr_SetString(PyExc_RuntimeError, "tensor_t.shape or tensor_t.stride allocation failed!");
      return NULL;
    }
    for(size_t i = 0; i < tz->ndim; i++){
      tz->shape[i] = tx->shape[i];
      tz->stride[i] = tx->stride[i];
    }
  }else{
    tz->shape = NULL;
    tz->stride = NULL;
  }
  tz->storage = tt_malloc(sizeof(storage_t));
  if(!tz->storage){
    destroy(tz);
    PyErr_SetString(PyExc_RuntimeError, "tensor_t.storage_t allocation failed!");
    return NULL;
  }
  tz->storage->bytes = tz->size * tz->element_size;
  tz->storage->device = tz->device;
  tz->storage->refcount = 1;
  tz->storage->ptr = tt_malloc(tz->storage->bytes);
  if(!tz->storage->ptr){
    destroy(tz);
    PyErr_SetString(PyExc_RuntimeError, "tensor_t.storage_t.ptr allocation failed!");
    return NULL;
  }
  tz->buf = tz->storage->ptr;
  return __cbrt_tensor__(tx, tz);
}

static PyObject *__floor_tensor__(const tensor_t *tx, tensor_t *tz){
  size_t N = tx->size;
  switch(tx->dtype){
    case FP16: {
      float16 *in = (float16 *)tx->buf;
      float16 *out = (float16 *)tz->buf;
      for(size_t i = 0; i < N; i++){
        float32 tmp = fp16_to_float(in[i]);
        out[i] = float_to_fp16(floorf(tmp));
      }
      break;
    }
    case FP32: {
      float32 *in = (float32 *)tx->buf;
      float32 *out = (float32 *)tz->buf;
      for(size_t i = 0; i < N; i++){ out[i] = floorf(in[i]); }
      break;
    }
    case FP64: {
      float64 *in = (float64 *)tx->buf;
      float64 *out = (float64 *)tz->buf;
      for(size_t i = 0; i < N; i++){ out[i] = floor(in[i]); }
      break;
    }
    case INT8:
    case UINT8:
    case INT16:
    case UINT16:
    case INT32:
    case UINT32:
    case INT64:
    case UINT64: { memcpy(tz->buf, tx->buf, N * tx->element_size); break; }
    default:
      PyErr_SetString(PyExc_TypeError, "Unsupported dtype for floor_()");
      return NULL;
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *floor_(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "operands must be the tensor capsule");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "floor_() is not supported on complex dtype tensor(s)");
    return NULL;
  }
  tensor_t *tz = alloc_result_tensor(tx);
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed");
    return NULL;
  }
  return __floor_tensor__(tx, tz);
}


static PyObject *__ceil_tensor__(const tensor_t *tx, tensor_t *tz){
  size_t N = tx->size;
  switch(tx->dtype){
    case FP16: {
      float16 *in = (float16 *)tx->buf;
      float16 *out = (float16 *)tz->buf;
      for(size_t i = 0; i < N; i++){
        float32 tmp = fp16_to_float(in[i]);
        out[i] = float_to_fp16(ceilf(tmp));
      }
      break;
    }
    case FP32: {
      float32 *in = (float32 *)tx->buf;
      float32 *out = (float32 *)tz->buf;
      for(size_t i = 0; i < N; i++){ out[i] = ceilf(in[i]); }
      break;
    }
    case FP64: {
      float64 *in = (float64 *)tx->buf;
      float64 *out = (float64 *)tz->buf;
      for(size_t i = 0; i < N; i++){ out[i] = ceil(in[i]); }
      break;
    }
    case INT8:
    case UINT8:
    case INT16:
    case UINT16:
    case INT32:
    case UINT32:
    case INT64:
    case UINT64: { memcpy(tz->buf, tx->buf, N * tx->element_size); break; }
    default:
      PyErr_SetString(PyExc_TypeError, "Unsupported dtype for ceil_()");
      return NULL;
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *ceil_(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "operands must be the tensor capsule");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "ceil_() is not supported on complex dtype tensor(s)");
    return NULL;
  }
  tensor_t *tz = alloc_result_tensor(tx);
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed");
    return NULL;
  }
  return __ceil_tensor__(tx, tz);
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
    PyErr_Format(PyExc_TypeError, "%% opertaion is not supported on complex dtype tensor(s)");
    return NULL;
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
  tensor_t *tz = tt_malloc(sizeof(tensor_t));
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
    tz->shape = tt_malloc(sizeof(size_t) * tz->ndim);
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
  tz->storage = tt_malloc(sizeof(storage_t));
  if(!tz->storage){
    if(tz->shape) free(tz->shape);
    free(tz);
    PyErr_SetString(PyExc_RuntimeError, "tensor_t.storage allocation failed!");
    return NULL;
  }
  tz->storage->bytes = tz->size * tz->element_size;
  tz->storage->device = tz->device;
  tz->storage->refcount = 1;
  tz->storage->ptr = tt_malloc(tz->storage->bytes);
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
  tensor_t *tz = tt_malloc(sizeof(tensor_t));
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
    tz->shape = tt_malloc(sizeof(size_t) * tz->ndim);
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
  tz->storage = tt_malloc(sizeof(storage_t));
  if(!tz->storage){
    if(tz->shape) free(tz->shape);
    free(tz);
    PyErr_SetString(PyExc_RuntimeError, "tensor_t.storage allocation failed!");
    return NULL;
  }
  tz->storage->bytes = tz->size * tz->element_size;
  tz->storage->device = tz->device;
  tz->storage->refcount = 1;
  tz->storage->ptr = tt_malloc(tz->storage->bytes);
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
    PyErr_SetString(PyExc_RuntimeError, "> operation is not supported on complex dtype tensor(s)");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_RuntimeError, "Internal error: size mismatch after broadcasting");
    return NULL;
  }
  tensor_t *tz = tt_malloc(sizeof(tensor_t));
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
    tz->shape = tt_malloc(sizeof(size_t) * tz->ndim);
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
  tz->storage = tt_malloc(sizeof(storage_t));
  if(!tz->storage){
    if(tz->shape) free(tz->shape);
    free(tz);
    PyErr_SetString(PyExc_RuntimeError, "tensor_t.storage allocation failed!");
    return NULL;
  }
  tz->storage->bytes = tz->size * tz->element_size;
  tz->storage->device = tz->device;
  tz->storage->refcount = 1;
  tz->storage->ptr = tt_malloc(tz->storage->bytes);
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
    PyErr_SetString(PyExc_RuntimeError, ">= operation is not supported on complex dtype tensor(s)");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_RuntimeError, "Internal error: size mismatch after broadcasting");
    return NULL;
  }
  tensor_t *tz = tt_malloc(sizeof(tensor_t));
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
    tz->shape = tt_malloc(sizeof(size_t) * tz->ndim);
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
  tz->storage = tt_malloc(sizeof(storage_t));
  if(!tz->storage){
    if(tz->shape) free(tz->shape);
    free(tz);
    PyErr_SetString(PyExc_RuntimeError, "tensor_t.storage allocation failed!");
    return NULL;
  }
  tz->storage->bytes = tz->size * tz->element_size;
  tz->storage->device = tz->device;
  tz->storage->refcount = 1;
  tz->storage->ptr = tt_malloc(tz->storage->bytes);
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
    PyErr_SetString(PyExc_RuntimeError, "< operation is not supported on complex dtype tensor(s)");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_RuntimeError, "Internal error: size mismatch after broadcasting");
    return NULL;
  }
  tensor_t *tz = tt_malloc(sizeof(tensor_t));
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
    tz->shape = tt_malloc(sizeof(size_t) * tz->ndim);
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
  tz->storage = tt_malloc(sizeof(storage_t));
  if(!tz->storage){
    if(tz->shape) free(tz->shape);
    free(tz);
    PyErr_SetString(PyExc_RuntimeError, "tensor_t.storage allocation failed!");
    return NULL;
  }
  tz->storage->bytes = tz->size * tz->element_size;
  tz->storage->device = tz->device;
  tz->storage->refcount = 1;
  tz->storage->ptr = tt_malloc(tz->storage->bytes);
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
    PyErr_SetString(PyExc_RuntimeError, "<= operation is not supported on complex dtype tensor(s)");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_RuntimeError, "Internal error: size mismatch after broadcasting");
    return NULL;
  }
  tensor_t *tz = tt_malloc(sizeof(tensor_t));
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
    tz->shape = tt_malloc(sizeof(size_t) * tz->ndim);
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
  tz->storage = tt_malloc(sizeof(storage_t));
  if(!tz->storage){
    if(tz->shape) free(tz->shape);
    free(tz);
    PyErr_SetString(PyExc_RuntimeError, "tensor_t.storage allocation failed!");
    return NULL;
  }
  tz->storage->bytes = tz->size * tz->element_size;
  tz->storage->device = tz->device;
  tz->storage->refcount = 1;
  tz->storage->ptr = tt_malloc(tz->storage->bytes);
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
    PyErr_SetString(PyExc_RuntimeError, "Negation operation on bool tensor is not supported");
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
    PyErr_SetString(PyExc_RuntimeError, "Negation operation on bool tensor is not supported");
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
    tensor_t *tz = tt_malloc(sizeof(tensor_t));
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
      tz->shape = tt_malloc(sizeof(size_t) * tz->ndim);
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
    tz->storage = tt_malloc(sizeof(storage_t));
    if(!tz->storage){
      if(tz->shape) free(tz->shape);
      free(tz);
      PyErr_SetString(PyExc_RuntimeError, "tensor_t.storage allocation failed!");
      return NULL;
    }
    tz->storage->bytes = tz->size * tz->element_size;
    tz->storage->device = tz->device;
    tz->storage->refcount = 1;
    tz->storage->ptr = tt_malloc(tz->storage->bytes);
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
    PyErr_SetString(PyExc_RuntimeError, "'>>' operation is not supported for floating points dtype tensor(s)");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "'>>' operation is not supported on complex dtype tensor(s)");
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
    PyErr_SetString(PyExc_RuntimeError, "'<<' operation is not supported for floating points dtype tensor(s)");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "'<<' operation is not supported on complex dtype tensor(s)");
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

// bitwise and
static PyObject *bitwise_and(PyObject *self, PyObject *args){
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
    PyErr_SetString(PyExc_TypeError, "tensor_t must have same dtype");
    return NULL;
  }
  if(tx->dtype == FP32 || tx->dtype == FP64){
    PyErr_SetString(PyExc_RuntimeError, "bitwise_and() operation is not supported on floating points dtype tensor(s)");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "bitwise_and() operation is not supported on complex dtype tensor(s)");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_RuntimeError, "Internal error: size mismatch after broadcasting");
    return NULL;
  }
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __bitwise_and_tensor__(tx, ty, tz);
}

// logical and
static PyObject *logical_and(PyObject *self, PyObject *args){
  PyObject *x, *y;
  if(!PyArg_ParseTuple(args, "OO", &x, &y)) return NULL;
  if(!PyCapsule_CheckExact(x) || !PyCapsule_CheckExact(y)){
    PyErr_SetString(PyExc_RuntimeError, "operands must be the tensor capsule");
    return NULL;
  }
  tensor_t *tx = (tensor_t *)PyCapsule_GetPointer(x, "tensor_t on CPU");
  tensor_t *ty = (tensor_t *)PyCapsule_GetPointer(y, "tensor_t on CPU");
  if(!tx || !ty){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  if(tx->dtype != ty->dtype){
    PyErr_SetString(PyExc_TypeError, "Both tensor_t(s) must have the same dtype");
    return NULL;
  }
  if(tx->device.type != ty->device.type || tx->device.type != CPU){
    PyErr_SetString(PyExc_RuntimeError, "Both tensor_t(s) must be on same device");
    return NULL;
  }
  tensor_t *tz = (tensor_t *)tt_malloc(sizeof(tensor_t));
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
    return NULL;
  }
  tz->dtype = (dtype_t)UINT8;
  tz->element_size = getsize(tz->dtype);
  tz->size = tx->size;
  tz->ndim = tx->ndim;
  tz->device = (device_t){CPU, 0};
  if(tx->ndim > 0 && tx->shape){
    tz->shape = (size_t *)tt_malloc(sizeof(size_t) * tz->ndim);
    tz->stride = (size_t *)tt_malloc(sizeof(size_t) * tz->ndim);
    if(!tz->shape || !tz->stride){
      free(tz);
      PyErr_SetString(PyExc_RuntimeError, "tensor_t.shape or tensor_t.stride allocation failed!");
      return NULL;
    }
    for(size_t i = 0; i < tz->ndim; i++){
      tz->shape[i] = tx->shape[i];
      tz->stride[i] = tx->stride[i];
    }
  }else{
    tz->shape = NULL;
    tz->stride = NULL;
  }
  tz->storage = (storage_t *)tt_malloc(sizeof(storage_t));
  if(!tz->storage){
    if(tz->shape) free(tz->shape);
    free(tz);
    PyErr_SetString(PyExc_RuntimeError, "tensor_t.storage allocation failed!");
    return NULL;
  }
  tz->storage->bytes = tz->size * tz->element_size;
  tz->storage->device = tz->device;
  tz->storage->refcount = 1;
  tz->storage->ptr = tt_malloc(tx->storage->bytes);
  if(!tz->storage->ptr){
    if(tz->shape) free(tz->shape);
    free(tz->storage);
    free(tz);
    PyErr_Format(PyExc_RuntimeError, "tensor_t.storage_t.ptr allocation failed");
    return NULL;
  }
  tz->buf = tz->storage->ptr;
  return __logical_and_tensor__(tx, ty, tz);
}

// bitwise nand
static PyObject *bitwise_nand(PyObject *self, PyObject *args){
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
    PyErr_SetString(PyExc_TypeError, "tensor_t must have same dtype");
    return NULL;
  }
  if(tx->dtype == FP32 || tx->dtype == FP64){
    PyErr_SetString(PyExc_RuntimeError, "bitwise_nand() operation is not supported on floating points dtype tensor(s)");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "bitwise_nand() operation is not supported on complex dtype tensor(s)");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_RuntimeError, "Internal error: size mismatch after broadcasting");
    return NULL;
  }
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __bitwise_nand_tensor__(tx, ty, tz);
}

// logical nand
static PyObject *logical_nand(PyObject *self, PyObject *args){
  PyObject *x, *y;
  if(!PyArg_ParseTuple(args, "OO", &x, &y)) return NULL;
  if(!PyCapsule_CheckExact(x) || !PyCapsule_CheckExact(y)){
    PyErr_SetString(PyExc_RuntimeError, "operands must be tensor capsules"); return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  tensor_t *ty = PyCapsule_GetPointer(y, "tensor_t on CPU");
  if(!tx || !ty){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule"); return NULL;
  }
  if(tx->device.type != CPU || ty->device.type != CPU){
    PyErr_SetString(PyExc_RuntimeError, "Only CPU tensors supported"); return NULL;
  }
  if(tx->dtype != ty->dtype){
    PyErr_SetString(PyExc_TypeError, "tensor_t must have same dtype"); return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_RuntimeError, "Size mismatch"); return NULL;
  }
  tensor_t *tz = alloc_result_tensor(tx);
  tz->dtype = BOOL;
  tz->element_size = getsize(tz->dtype);
  return __logical_nand_tensor__(tx, ty, tz);
}

// bitwise or
static PyObject *bitwise_or(PyObject *self, PyObject *args){
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
    PyErr_SetString(PyExc_TypeError, "tensor_t must have same dtype");
    return NULL;
  }
  if(tx->dtype == FP32 || tx->dtype == FP64){
    PyErr_SetString(PyExc_RuntimeError, "bitwise_or() operation is not supported on floating points dtype tensor(s)");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "bitwise_or() operation is not supported on complex dtype tensor(s)");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_RuntimeError, "Internal error: size mismatch after broadcasting");
    return NULL;
  }
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __bitwise_or_tensor__(tx, ty, tz);
}

// logical or
static PyObject *logical_or(PyObject *self, PyObject *args){
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
    PyErr_SetString(PyExc_TypeError, "tensor_t must have same dtype");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_RuntimeError, "Size mismatch");
    return NULL;
  }
  tensor_t *tz = alloc_result_tensor(tx);
  tz->dtype = BOOL;
  tz->element_size = getsize(tz->dtype);
  return __logical_or_tensor__(tx, ty, tz);
}

// bitwise nor
static PyObject *bitwise_nor(PyObject *self, PyObject *args){
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
    PyErr_SetString(PyExc_TypeError, "tensor_t musr have same dtype");
    return NULL;
  }
  if(tx->dtype == FP32 || tx->dtype == FP64){
    PyErr_SetString(PyExc_RuntimeError, "bitwise_nor() operation is not supported on floating points dtype tensor(s)");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "bitwise_nor() operation is not supported on complex dtype tensor(s)");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_RuntimeError, "Internal error: size mismatch after broadcasting");
    return NULL;
  }
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __bitwise_nor_tensor__(tx, ty, tz);
}

// logical nor
static PyObject *logical_nor(PyObject *self, PyObject *args){
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
    PyErr_SetString(PyExc_TypeError, "tensor_t must have same dtype");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_RuntimeError, "Size mismatch");
    return NULL;
  }
  tensor_t *tz = alloc_result_tensor(tx);
  tz->dtype = BOOL;
  tz->element_size = getsize(tz->dtype);
  return __logical_nor_tensor__(tx, ty, tz);
}

// bitwise not
static PyObject *bitwise_not(PyObject *self, PyObject *args){
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
    PyErr_SetString(PyExc_RuntimeError, "bitwise_not() operation is not supported on floating points dtype tensor(s)");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "bitwise_not() operation is not supported on complex dtype tensor(s)");
    return NULL;
  }
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __bitwise_not_tensor__(tx, tz);
}

// logical not
static PyObject *logical_not(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "operand must be tensor capsule");
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
  tensor_t *tz = alloc_result_tensor(tx);
  tz->dtype = BOOL;
  tz->element_size = getsize(tz->dtype);
  return __logical_not_tensor__(tx, tz);
}

// bitwise xor
static PyObject *bitwise_xor(PyObject *self, PyObject *args){
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
    PyErr_SetString(PyExc_RuntimeError, "bitwise_xnor() operation is not supported on floating points dtype tensor(s)");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "bitwise_xor() operation is not supported on complex dtype tensor");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_RuntimeError, "Internal error: size mismatch after broadcasting");
    return NULL;
  }
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __bitwise_xor_tensor__(tx, ty, tz);
}

// logical xor
static PyObject *logical_xor(PyObject *self, PyObject *args){
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
    PyErr_SetString(PyExc_TypeError, "tensor_t must have same dtype");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_RuntimeError, "Size mismatch");
    return NULL;
  }
  tensor_t *tz = alloc_result_tensor(tx);
  tz->dtype = BOOL;
  tz->element_size = getsize(tz->dtype);
  return __logical_xor_tensor__(tx, ty, tz);
}

// bitwise xnor
static PyObject *bitwise_xnor(PyObject *self, PyObject *args){
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
    PyErr_SetString(PyExc_RuntimeError, "bitwise_xnor() operation is not supported on floating points dtype tensor(s)");
    return NULL;
  }
  if(tx->dtype == CMPX64 || tx->dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "bitwise_xnor() operation is not supported on complex dtype tensor(s)");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_RuntimeError, "Internal error: size mismatch after broadcasting");
    return NULL;
  }
  tensor_t *tz = NULL;
  tz = alloc_result_tensor(tx);
  return __bitwise_xnor_tensor__(tx, ty, tz);
}

// logical xnor
static PyObject *logical_xnor(PyObject *self, PyObject *args){
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
    PyErr_SetString(PyExc_TypeError, "tensor_t must have same dtype");
    return NULL;
  }
  if(tx->size != ty->size){
    PyErr_SetString(PyExc_RuntimeError, "Size mismatch");
    return NULL;
  }
  tensor_t *tz = alloc_result_tensor(tx);
  tz->dtype = BOOL;
  tz->element_size = getsize(tz->dtype);
  return __logical_xnor_tensor__(tx, ty, tz);
}

SUM_AXIS_KERNEL(int8, int8, int32);
SUM_AXIS_KERNEL(uint8, uint8, uint64);
SUM_AXIS_KERNEL(int16, int16, int32);
SUM_AXIS_KERNEL(uint16, uint16, uint64);
SUM_AXIS_KERNEL(int32, int32, int64);
SUM_AXIS_KERNEL(uint32, uint32, uint64);
SUM_AXIS_KERNEL(int64, int64, int64);
SUM_AXIS_KERNEL(uint64, uint64, uint64);
SUM_AXIS_FP16_KERNEL(float16, float32);
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
    float32 real_sum = 0.0f;
    float32 imag_sum = 0.0f;
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
    float64 real_sum = 0.0;
    float64 imag_sum = 0.0;
    for(int r = 0; r < reduced_dim; r++){
      idx_in[axis] = r;
      int in_offset = 0;
      for(int k = 0; k < tx->ndim; k++){
        in_offset += idx_in[k] * tx->stride[k];
      }
      real_sum += (float64)in[in_offset].real;
      imag_sum += (float64)in[in_offset].imag;
    }
    out[out_index].real = (float64)real_sum;
    out[out_index].imag = (float64)imag_sum;
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
      case FP16: return __sum_axis_float16__(tx, tz, axis);
      case FP32: return __sum_axis_float32__(tx, tz, axis);
      case FP64: return __sum_axis_float64__(tx, tz, axis);
      case CMPX64: return __sum_axis_cmpx64__(tx, tz, axis);
      case CMPX128: return __sum_axis_cmpx128__(tx, tz, axis);
      case ERROR: {
        destroy(tz);
        PyErr_SetString(PyExc_RuntimeError, "something went wrong");
        return NULL;
      }
      default: {
        destroy(tz);
        PyErr_SetString(PyExc_TypeError, "sum not supported for this dtype");
        return NULL;
      }
    }
  }
  tensor_t *tz = tt_malloc(sizeof(tensor_t));
  if(!tz){
    PyErr_SetString(PyExc_MemoryError, "tensor allocation failed");
    return NULL;
  }
  tz->dtype = tx->dtype;
  tz->device = tx->device;
  tz->element_size = tx->element_size;
  tz->ndim = out_ndim;
  tz->shape = tt_malloc(sizeof(size_t) * out_ndim);
  tz->stride = tt_malloc(sizeof(size_t) * out_ndim);
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
  tz->storage = tt_malloc(sizeof(storage_t));
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
      PyErr_SetString(PyExc_RuntimeError, "something went wrong");
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
  tensor_t *tz = tt_malloc(sizeof(tensor_t));
  if(!tz){
    PyErr_SetString(PyExc_MemoryError, "tensor allocation failed");
    return NULL;
  }
  tz->dtype = tx->dtype;
  tz->device = tx->device;
  tz->element_size = tx->element_size;
  tz->ndim = tx->ndim;
  tz->shape = tt_malloc(sizeof(size_t) * tz->ndim);
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
  tz->stride = tt_malloc(sizeof(size_t) * tz->ndim);
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
  tz->storage = tt_malloc(sizeof(storage_t));
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
    case FP16: {
      float16_t *a = (float16_t *)tx->buf;
      float16_t *b = (float16_t *)ty->buf;
      float16_t *c = (float16_t *)tz->buf;
      size_t batch_size = 1;
      for(size_t i = 0; i < tx->ndim - 2; i++){
        batch_size *= tx->shape[i];
      }
      for(size_t batch = 0; batch < batch_size; batch++){
        size_t a_offset = batch * m * k;
        size_t b_offset = batch * k * n;
        size_t c_offset = batch * m * n;
        for(size_t i = 0; i < m; i++){
          for(size_t j = 0; j < n; j++){
            float32 acc = 0.0f;
            for(size_t kk = 0; kk < k; kk++){
              float32 ax = fp16_to_float(a[a_offset + i * k + kk]);
              float32 by = fp16_to_float(b[b_offset + kk * n + j]);
              acc += ax * by;
            }
            c[c_offset + i * n + j] = float_to_fp16(acc);
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

static PyObject *conj_(PyObject *self, PyObject *args){
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
  tensor_t *tz = tt_malloc(sizeof(tensor_t));
  tz->dtype = (tx->dtype == CMPX64) ? CMPX64 : CMPX128;
  tz->element_size = getsize(tz->dtype);
  tz->device = (device_t){CPU, 0};
  tz->size = 1;
  tz->storage = tt_malloc(sizeof(storage_t));
  tz->storage->device = tz->device;
  tz->storage->refcount = 1;
  tz->storage->bytes = tz->element_size;
  tz->storage->ptr = tt_malloc(tz->storage->bytes);
  tz->buf = tz->storage->ptr;
  if(tx->ndim == 0){
    tz->ndim = 0;
    tz->shape = NULL;
    tz->stride = NULL;
    if(tx->dtype == CMPX64){
      ((float*)tz->buf)[0] = ((complex64*)tx->buf)[0].real;
    }else{
      ((float64*)tz->buf)[0] = ((complex128*)tx->buf)[0].real;
    }
    return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
  }
  tz->ndim = tx->ndim;
  tz->shape = tt_malloc(sizeof(size_t) * tz->ndim);
  tz->stride = tt_malloc(sizeof(size_t) * tz->ndim);
  for(size_t i = 0; i < tz->ndim; i++){
    tz->shape[i] = tx->shape[i];
    tz->stride[i] = tx->stride[i];
  }
  tz->size = tx->size;
  tz->storage->bytes = tz->size * tz->element_size;
  tz->storage->ptr = tt_malloc(tz->storage->bytes);
  tz->buf = tz->storage->ptr;
  if(tx->dtype == CMPX64){
    complex64 *in = (complex64 *)tx->buf;
    complex64 *out = (complex64 *)tz->buf;
    for(size_t i = 0; i < tz->size; i++){
      out[i].real = in[i].real;
      out[i].imag = -in[i].imag;
    }
  }else if(tx->dtype == CMPX128){
    complex128 *in = (complex128 *)tx->buf;
    complex128 *out = (complex128 *)tz->buf;
    for(size_t i = 0; i < tz->size; i++){
      out[i].real = in[i].real;
      out[i].imag = -in[i].imag;
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
  tensor_t *tz = tt_malloc(sizeof(tensor_t));
  tz->dtype = (tx->dtype == CMPX64) ? FP32 : FP64;
  tz->element_size = getsize(tz->dtype);
  tz->device = (device_t){CPU, 0};
  tz->size = 1;
  tz->storage = tt_malloc(sizeof(storage_t));
  tz->storage->device = tz->device;
  tz->storage->refcount = 1;
  tz->storage->bytes = tz->element_size;
  tz->storage->ptr = tt_malloc(tz->storage->bytes);
  tz->buf = tz->storage->ptr;
  if(tx->ndim == 0){
    tz->ndim = 0;
    tz->shape = NULL;
    tz->stride = NULL;
    if(tx->dtype == CMPX64){
      ((float*)tz->buf)[0] = ((complex64*)tx->buf)[0].real;
    }else{
      ((float64*)tz->buf)[0] = ((complex128*)tx->buf)[0].real;
    }
    return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
  }
  tz->ndim = tx->ndim;
  tz->shape = tt_malloc(sizeof(size_t) * tz->ndim);
  tz->stride = tt_malloc(sizeof(size_t) * tz->ndim);
  for(size_t i = 0; i < tz->ndim; i++){
    tz->shape[i] = tx->shape[i];
    tz->stride[i] = tx->stride[i];
  }
  tz->size = tx->size;
  tz->storage->bytes = tz->size * tz->element_size;
  tz->storage->ptr = tt_malloc(tz->storage->bytes);
  tz->buf = tz->storage->ptr;
  if(tx->dtype == CMPX64){
    complex64 *src = (complex64*)tx->buf;
    float32 *dst = (float*)tz->buf;
    for(size_t i = 0; i < tx->size; i++) dst[i] = src[i].real;
  }else{
    complex128 *src = (complex128*)tx->buf;
    float64 *dst = (float64*)tz->buf;
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
  tensor_t *tz = tt_malloc(sizeof(tensor_t));
  tz->dtype = (tx->dtype == CMPX64) ? FP32 : FP64;
  tz->element_size = getsize(tz->dtype);
  tz->device = (device_t){CPU, 0};
  tz->size = 1;
  tz->storage = tt_malloc(sizeof(storage_t));
  tz->storage->device = tz->device;
  tz->storage->refcount = 1;
  tz->storage->bytes = tz->element_size;
  tz->storage->ptr = tt_malloc(tz->storage->bytes);
  tz->buf = tz->storage->ptr;
  if(tx->ndim == 0){
    tz->ndim = 0;
    tz->shape = NULL;
    tz->stride = NULL;
    if(tx->dtype == CMPX64){
      ((float*)tz->buf)[0] = ((complex64*)tx->buf)[0].imag;
    }else{
      ((float64*)tz->buf)[0] = ((complex128*)tx->buf)[0].imag;
    }
    return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
  }
  tz->ndim = tx->ndim;
  tz->shape = tt_malloc(sizeof(size_t) * tz->ndim);
  tz->stride = tt_malloc(sizeof(size_t) * tz->ndim);
  for(size_t i = 0; i < tz->ndim; i++){
    tz->shape[i] = tx->shape[i];
    tz->stride[i] = tx->stride[i];
  }
  tz->size = tx->size;
  tz->storage->bytes = tz->size * tz->element_size;
  tz->storage->ptr = tt_malloc(tz->storage->bytes);
  tz->buf = tz->storage->ptr;
  if(tx->dtype == CMPX64){
    complex64 *src = (complex64*)tx->buf;
    float32 *dst = (float*)tz->buf;
    for(size_t i = 0; i < tx->size; i++) dst[i] = src[i].imag;
  }else{
    complex128 *src = (complex128*)tx->buf;
    float64 *dst = (float64*)tz->buf;
    for(size_t i = 0; i < tx->size; i++) dst[i] = src[i].imag;
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

tensor_t *tensor_empty_like(tensor_t *tx, dtype_t dtype){
  tensor_t *tz = tt_malloc(sizeof(tensor_t));
  if(!tz){
    PyErr_SetString(PyExc_MemoryError, "tensor_t allocation failed");
    return NULL;
  }
  tz->dtype = dtype;
  tz->element_size = getsize(dtype);
  tz->device = tx->device;
  tz->ndim = tx->ndim;
  tz->size = tx->size;
  tz->shape = tt_malloc(sizeof(size_t) * tz->ndim);
  if(!tz->shape){
    PyErr_SetString(PyExc_MemoryError, "shape allocation failed");
    free(tz);
    return NULL;
  }
  tz->stride = tt_malloc(sizeof(size_t) * tz->ndim);
  if(!tz->stride){
    PyErr_SetString(PyExc_MemoryError, "stride allocation failed");
    free(tz->shape);
    free(tz);
    return NULL;
  }
  for(size_t i = 0; i < tz->ndim; i++){
    tz->shape[i] = tx->shape[i];
    tz->stride[i] = tx->stride[i];
  }
  tz->storage = tt_malloc(sizeof(storage_t));
  if(!tz->storage){
    PyErr_SetString(PyExc_MemoryError, "storage allocation failed");
    free(tz->stride);
    free(tz->shape);
    free(tz);
    return NULL;
  }
  tz->storage->device = tz->device;
  tz->storage->refcount = 1;
  tz->storage->bytes = tz->size * tz->element_size;
  tz->storage->ptr = tt_malloc(tz->storage->bytes);
  if(!tz->storage->ptr){
    PyErr_SetString(PyExc_MemoryError, "buffer allocation failed");
    free(tz->storage);
    free(tz->stride);
    free(tz->shape);
    free(tz);
    return NULL;
  }
  tz->buf = tz->storage->ptr;
  return tz;
}

static PyObject *exp_(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor_t capsule pointer");
    return NULL;
  }
  dtype_t out_dtype;
  switch(tx->dtype){
    case INT8: case UINT8:
    case INT16: case UINT16:
    case INT32: case UINT32: out_dtype = FP32; break;
    case INT64: case UINT64: out_dtype = FP64; break;
    case FP16: out_dtype = FP32; break;
    case FP32: out_dtype = FP32; break;
    case FP64: out_dtype = FP64; break;
    case CMPX64: out_dtype = CMPX64; break;
    case CMPX128: out_dtype = CMPX128; break;
    default: PyErr_SetString(PyExc_TypeError, "exp_() unsupported dtype"); return NULL;
  }
  tensor_t *tz = tensor_empty_like(tx, out_dtype);
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "Failed to allocate output tensor");
    return NULL;
  }
  size_t N = tx->size;
  if(out_dtype == FP32){
    float32 *out = (float32*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 v;
      switch(tx->dtype){
        case INT8: v = (float)((int8*)tx->buf)[i]; break;
        case UINT8: v = (float)((uint8*)tx->buf)[i]; break;
        case INT16: v = (float)((int16*)tx->buf)[i]; break;
        case UINT16: v = (float)((uint16*)tx->buf)[i]; break;
        case INT32: v = (float)((int32*)tx->buf)[i]; break;
        case UINT32: v = (float)((uint32*)tx->buf)[i]; break;
        case FP16: v = fp16_to_float(((float16 *)tx->buf)[i]); break;
        case FP32: v = ((float32*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "exp_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = expf(v);
    }
  }
  else if(out_dtype == FP64){
    float64 *out = (float64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 v;
      switch(tx->dtype){
        case INT64: v = (float64)((int64*)tx->buf)[i]; break;
        case UINT64: v = (float64)((uint64*)tx->buf)[i]; break;
        case FP64: v = ((float64*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "exp_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = exp(v);
    }
  }else if(out_dtype == CMPX64){
    complex64 *in = (complex64 *)tx->buf;
    complex64 *out = (complex64 *)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 a = in[i].real;
      float32 b = in[i].imag;
      float32 expa = expf(a);
      float32 cosb = cosf(b);
      float32 sinb = sinf(b);
      out[i].real = expa * cosb;
      out[i].imag = expa * sinb;
    }
  }else if(out_dtype == CMPX128){
    complex128 *in = (complex128 *)tx->buf;
    complex128 *out = (complex128 *)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 a = in[i].real;
      float64 b = in[i].imag;
      float64 expa = exp(a);
      float64 cosb = cos(b);
      float64 sinb = sin(b);
      out[i].real = expa * cosb;
      out[i].imag = expa * sinb;
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *log_(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor_t capsule pointer");
    return NULL;
  }
  dtype_t out_dtype;
  switch(tx->dtype){
    case INT8: case UINT8:
    case INT16: case UINT16:
    case INT32: case UINT32: out_dtype = FP32; break;
    case INT64: case UINT64: out_dtype = FP64; break;
    case FP16: out_dtype = FP32; break;
    case FP32: out_dtype = FP32; break;
    case FP64: out_dtype = FP64; break;
    case CMPX64: out_dtype = CMPX64; break;
    case CMPX128: out_dtype = CMPX128; break;
    default: PyErr_SetString(PyExc_TypeError, "log_() unsupported dtype"); return NULL;
  }
  tensor_t *tz = tensor_empty_like(tx, out_dtype);
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "Failed to allocate output tensor");
    return NULL;
  }
  size_t N = tx->size;
  if(out_dtype == FP32){
    float32 *out = (float32*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 v;
      switch(tx->dtype){
        case INT8: v = (float32)((int8*)tx->buf)[i]; break;
        case UINT8: v = (float32)((uint8*)tx->buf)[i]; break;
        case INT16: v = (float32)((int16*)tx->buf)[i]; break;
        case UINT16: v = (float32)((uint16*)tx->buf)[i]; break;
        case INT32: v = (float32)((int32*)tx->buf)[i]; break;
        case UINT32: v = (float32)((uint32*)tx->buf)[i]; break;
        case FP16: v = fp16_to_float(((float16 *)tx->buf)[i]); break;
        case FP32: v = ((float32*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "log_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = logf(v);
    }
  }else if(out_dtype == FP64){
    float64 *out = (float64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 v;
      switch(tx->dtype){
        case INT64: v = (float64)((int64*)tx->buf)[i]; break;
        case UINT64: v = (float64)((uint64*)tx->buf)[i]; break;
        case FP64: v = ((float64*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "log_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = log(v);
    }
  }else if(out_dtype == CMPX64){
    complex64 *in = (complex64 *)tx->buf;
    complex64 *out = (complex64 *)tz->buf;
    for(size_t i = 0; i < N; i++){
        float32 a = in[i].real;
        float32 b = in[i].imag;
        float32 mag = sqrtf(a*a + b*b);
        float32 angle = atan2f(b, a);
        out[i].real = logf(mag);
        out[i].imag = angle;
    }
  }else if(out_dtype == CMPX128){
    complex128 *in  = (complex128*)tx->buf;
    complex128 *out = (complex128*)tz->buf;
    for(size_t i = 0; i < N; i++){
        float64 a = in[i].real;
        float64 b = in[i].imag;
        float64 modulus = sqrt(a*a + b*b);
        float64 angle = atan2(b, a);
        out[i].real = log(modulus);
        out[i].imag = angle;
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *log2_(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor_t capsule pointer");
    return NULL;
  }
  dtype_t out_dtype;
  switch(tx->dtype){
    case INT8: case UINT8:
    case INT16: case UINT16:
    case INT32: case UINT32: out_dtype = FP32; break;
    case INT64: case UINT64: out_dtype = FP64; break;
    case FP16: out_dtype = FP32; break;
    case FP32: out_dtype = FP32; break;
    case FP64: out_dtype = FP64; break;
    case CMPX64: out_dtype = CMPX64; break;
    case CMPX128: out_dtype = CMPX128; break;
    default: PyErr_SetString(PyExc_TypeError, "log2_() unsupported dtype"); return NULL;
  }
  tensor_t *tz = tensor_empty_like(tx, out_dtype);
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "Failed to allocate output tensor");
    return NULL;
  }
  size_t N = tx->size;
  if(out_dtype == FP32){
    float32 *out = (float32*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 v;
      switch(tx->dtype){
        case INT8: v = (float32)((int8*)tx->buf)[i]; break;
        case UINT8: v = (float32)((uint8*)tx->buf)[i]; break;
        case INT16: v = (float32)((int16*)tx->buf)[i]; break;
        case UINT16: v = (float32)((uint16*)tx->buf)[i]; break;
        case INT32: v = (float32)((int32*)tx->buf)[i]; break;
        case UINT32: v = (float32)((uint32*)tx->buf)[i]; break;
        case FP16: v = fp16_to_float(((float16 *)tx->buf)[i]); break;
        case FP32: v = ((float32*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "log2_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = log2f(v);
    }
  }else if(out_dtype == FP64){
    float64 *out = (float64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 v;
      switch(tx->dtype){
        case INT64: v = (float64)((int64*)tx->buf)[i]; break;
        case UINT64: v = (float64)((uint64*)tx->buf)[i]; break;
        case FP64: v = ((float64*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "log2_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = log2(v);
    }
  }else if(out_dtype == CMPX64){
    float32 inv_ln2 = 1.0f / logf(2.0f);
    complex64 *in  = (complex64*)tx->buf;
    complex64 *out = (complex64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 a = in[i].real;
      float32 b = in[i].imag;
      float32 modulus = sqrtf(a*a + b*b);
      float32 angle = atan2f(b, a);
      out[i].real = logf(modulus) * inv_ln2;
      out[i].imag = angle * inv_ln2;
    }
  }else if(out_dtype == CMPX128){
    float64 inv_ln2 = 1.0 / log(2.0);
    complex128 *in  = (complex128*)tx->buf;
    complex128 *out = (complex128*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 a = in[i].real;
      float64 b = in[i].imag;
      float64 modulus = sqrt(a*a + b*b);
      float64 angle   = atan2(b, a);
      out[i].real = log(modulus) * inv_ln2;
      out[i].imag = angle * inv_ln2;
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *log10_(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor_t capsule pointer");
    return NULL;
  }
  dtype_t out_dtype;
  switch(tx->dtype){
    case INT8: case UINT8:
    case INT16: case UINT16:
    case INT32: case UINT32: out_dtype = FP32; break;
    case INT64: case UINT64: out_dtype = FP64; break;
    case FP16: out_dtype = FP32; break;
    case FP32: out_dtype = FP32; break;
    case FP64: out_dtype = FP64; break;
    case CMPX64: out_dtype = CMPX64; break;
    case CMPX128: out_dtype = CMPX128; break;
    default: PyErr_SetString(PyExc_TypeError, "log10_() unsupported dtype"); return NULL;
  }
  tensor_t *tz = tensor_empty_like(tx, out_dtype);
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "Failed to allocate output tensor");
    return NULL;
  }
  size_t N = tx->size;
  if(out_dtype == FP32){
    float32 *out = (float32*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 v;
      switch(tx->dtype){
        case INT8: v = (float)((int8*)tx->buf)[i]; break;
        case UINT8: v = (float)((uint8*)tx->buf)[i]; break;
        case INT16: v = (float)((int16*)tx->buf)[i]; break;
        case UINT16: v = (float)((uint16*)tx->buf)[i]; break;
        case INT32: v = (float)((int32*)tx->buf)[i]; break;
        case UINT32: v = (float)((uint32*)tx->buf)[i]; break;
        case FP16: v = fp16_to_float(((float16 *)tx->buf)[i]); break;
        case FP32: v = ((float32*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "log10_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = log10f(v);
    }
  }else if(out_dtype == FP64){
    float64 *out = (float64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 v;
      switch(tx->dtype){
        case INT64: v = (float64)((int64*)tx->buf)[i]; break;
        case UINT64: v = (float64)((uint64*)tx->buf)[i]; break;
        case FP64: v = ((float64*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "log10_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = log10(v);
    }
  }else if(out_dtype == CMPX64){
    float32 inv_ln10 = 1.0f / logf(10.0f);
    complex64 *in  = (complex64*)tx->buf;
    complex64 *out = (complex64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 a = in[i].real;
      float32 b = in[i].imag;
      float32 modulus = sqrtf(a*a + b*b);
      float32 angle   = atan2f(b, a);
      out[i].real = logf(modulus) * inv_ln10;
      out[i].imag = angle * inv_ln10;
    }
  }else if(out_dtype == CMPX128){
    float64 inv_ln10 = 1.0 / log(10.0);
    complex128 *in  = (complex128*)tx->buf;
    complex128 *out = (complex128*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 a = in[i].real;
      float64 b = in[i].imag;
      float64 modulus = sqrt(a*a + b*b);
      float64 angle   = atan2(b, a);
      out[i].real = log(modulus) * inv_ln10;
      out[i].imag = angle * inv_ln10;
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *sin_(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor_t capsule pointer");
    return NULL;
  }
  dtype_t out_dtype;
  switch(tx->dtype){
    case INT8: case UINT8:
    case INT16: case UINT16:
    case INT32: case UINT32: out_dtype = FP32; break;
    case INT64: case UINT64: out_dtype = FP64; break;
    case FP16: out_dtype = FP32; break;
    case FP32: out_dtype = FP32; break;
    case FP64: out_dtype = FP64; break;
    case CMPX64: out_dtype = CMPX64; break;
    case CMPX128: out_dtype = CMPX128; break;
    default: PyErr_SetString(PyExc_TypeError, "sin_() unsupported dtype"); return NULL;
  }
  tensor_t *tz = tensor_empty_like(tx, out_dtype);
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "Failed to allocate output tensor");
    return NULL;
  }
  size_t N = tx->size;
  if(out_dtype == FP32){
    float32 *out = (float32*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 v;
      switch(tx->dtype){
        case INT8: v = (float32)((int8*)tx->buf)[i]; break;
        case UINT8: v = (float32)((uint8*)tx->buf)[i]; break;
        case INT16: v = (float32)((int16*)tx->buf)[i]; break;
        case UINT16: v = (float32)((uint16*)tx->buf)[i]; break;
        case INT32: v = (float32)((int32*)tx->buf)[i]; break;
        case UINT32: v = (float32)((uint32*)tx->buf)[i]; break;
        case FP16: v = fp16_to_float(((float16 *)tx->buf)[i]); break;
        case FP32: v = ((float32*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "sin_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = sinf(v);
    }
  }else if(out_dtype == FP64){
    float64 *out = (float64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 v;
      switch(tx->dtype){
        case INT64: v = (float64)((int64*)tx->buf)[i]; break;
        case UINT64: v = (float64)((uint64*)tx->buf)[i]; break;
        case FP64: v = ((float64*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "sin_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = sin(v);
    }
  }else if(out_dtype == CMPX64){
    complex64 *in  = (complex64*)tx->buf;
    complex64 *out = (complex64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 a = in[i].real;
      float32 b = in[i].imag;
      out[i].real = sinf(a) * coshf(b);
      out[i].imag = cosf(a) * sinhf(b);
    }
  }else if(out_dtype == CMPX128){
    complex128 *in  = (complex128*)tx->buf;
    complex128 *out = (complex128*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 a = in[i].real;
      float64 b = in[i].imag;
      out[i].real = sin(a) * cosh(b);
      out[i].imag = cos(a) * sinh(b);
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *cos_(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor_t capsule pointer");
    return NULL;
  }
  dtype_t out_dtype;
  switch(tx->dtype){
    case INT8: case UINT8:
    case INT16: case UINT16:
    case INT32: case UINT32: out_dtype = FP32; break;
    case INT64: case UINT64: out_dtype = FP64; break;
    case FP16: out_dtype = FP32; break;
    case FP32: out_dtype = FP32; break;
    case FP64: out_dtype = FP64; break;
    case CMPX64: out_dtype = CMPX64; break;
    case CMPX128: out_dtype = CMPX128; break;
    default: PyErr_SetString(PyExc_TypeError, "cos_() unsupported dtype"); return NULL;
  }
  tensor_t *tz = tensor_empty_like(tx, out_dtype);
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "Failed to allocate output tensor");
    return NULL;
  }
  size_t N = tx->size;
  if(out_dtype == FP32){
    float32 *out = (float32*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 v;
      switch(tx->dtype){
        case INT8: v = (float)((int8*)tx->buf)[i]; break;
        case UINT8: v = (float)((uint8*)tx->buf)[i]; break;
        case INT16: v = (float)((int16*)tx->buf)[i]; break;
        case UINT16: v = (float)((uint16*)tx->buf)[i]; break;
        case INT32: v = (float)((int32*)tx->buf)[i]; break;
        case UINT32: v = (float)((uint32*)tx->buf)[i]; break;
        case FP16: v = fp16_to_float(((float16 *)tx->buf)[i]); break;
        case FP32: v = ((float32*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "cos_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = cosf(v);
    }
  }else if(out_dtype == FP64){
    float64 *out = (float64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 v;
      switch(tx->dtype){
        case INT64: v = (float64)((int64*)tx->buf)[i]; break;
        case UINT64: v = (float64)((uint64*)tx->buf)[i]; break;
        case FP64: v = ((float64*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "cos_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = cos(v);
    }
  }else if(out_dtype == CMPX64){
    complex64 *in  = (complex64*)tx->buf;
    complex64 *out = (complex64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 a = in[i].real;
      float32 b = in[i].imag;
      out[i].real = cosf(a) * coshf(b);
      out[i].imag = -sinf(a) * sinhf(b);
    }
  }else if(out_dtype == CMPX128){
    complex128 *in  = (complex128*)tx->buf;
    complex128 *out = (complex128*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 a = in[i].real;
      float64 b = in[i].imag;
      out[i].real = cos(a) * cosh(b);
      out[i].imag = -sin(a) * sinh(b);
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *tan_(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor_t capsule pointer");
    return NULL;
  }
  dtype_t out_dtype;
  switch(tx->dtype){
    case INT8: case UINT8:
    case INT16: case UINT16:
    case INT32: case UINT32: out_dtype = FP32; break;
    case INT64: case UINT64: out_dtype = FP64; break;
    case FP16: out_dtype = FP32; break;
    case FP32: out_dtype = FP32; break;
    case FP64: out_dtype = FP64; break;
    case CMPX64: out_dtype = CMPX64; break;
    case CMPX128: out_dtype = CMPX128; break;
    default: PyErr_SetString(PyExc_TypeError, "tan_() unsupported dtype"); return NULL;
  }
  tensor_t *tz = tensor_empty_like(tx, out_dtype);
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "Failed to allocate output tensor");
    return NULL;
  }
  size_t N = tx->size;
  if(out_dtype == FP32){
    float32 *out = (float32*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 v;
      switch(tx->dtype){
        case INT8: v = (float)((int8*)tx->buf)[i]; break;
        case UINT8: v = (float)((uint8*)tx->buf)[i]; break;
        case INT16: v = (float)((int16*)tx->buf)[i]; break;
        case UINT16: v = (float)((uint16*)tx->buf)[i]; break;
        case INT32: v = (float)((int32*)tx->buf)[i]; break;
        case UINT32: v = (float)((uint32*)tx->buf)[i]; break;
        case FP16: v = fp16_to_float(((float16 *)tx->buf)[i]); break;
        case FP32: v = ((float32*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "tan_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = tanf(v);
    }
  }else if(out_dtype == FP64){
    float64 *out = (float64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 v;
      switch(tx->dtype){
        case INT64: v = (float64)((int64*)tx->buf)[i]; break;
        case UINT64: v = (float64)((uint64*)tx->buf)[i]; break;
        case FP64: v = ((float64*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "tan_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = tan(v);
    }
  }else if(out_dtype == CMPX64){
    complex64 *in  = (complex64*)tx->buf;
    complex64 *out = (complex64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 a = in[i].real;
      float32 b = in[i].imag;
      out[i].real = sinf(a) * coshf(b);
      out[i].imag = cosf(a) * sinhf(b);
    }
  }else if(out_dtype == CMPX128){
    complex128 *in  = (complex128*)tx->buf;
    complex128 *out = (complex128*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 a = in[i].real;
      float64 b = in[i].imag;
      out[i].real = sinf(a) * coshf(b);
      out[i].imag = cosf(a) * sinhf(b);
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *asin_(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor_t capsule pointer");
    return NULL;
  }
  dtype_t out_dtype;
  switch(tx->dtype){
    case INT8: case UINT8:
    case INT16: case UINT16:
    case INT32: case UINT32: out_dtype = FP32; break;
    case INT64: case UINT64: out_dtype = FP64; break;
    case FP16: out_dtype = FP32; break;
    case FP32: out_dtype = FP32; break;
    case FP64: out_dtype = FP64; break;
    case CMPX64: out_dtype = CMPX64; break;
    case CMPX128: out_dtype = CMPX128; break;
    default: PyErr_SetString(PyExc_TypeError, "asin_() unsupported dtype"); return NULL;
  }
  tensor_t *tz = tensor_empty_like(tx, out_dtype);
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "Failed to allocate output tensor");
    return NULL;
  }
  size_t N = tx->size;
  if(out_dtype == FP32){
    float32 *out = (float32*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 v;
      switch(tx->dtype){
        case INT8: v = (float32)((int8*)tx->buf)[i]; break;
        case UINT8: v = (float32)((uint8*)tx->buf)[i]; break;
        case INT16: v = (float32)((int16*)tx->buf)[i]; break;
        case UINT16: v = (float32)((uint16*)tx->buf)[i]; break;
        case INT32: v = (float32)((int32*)tx->buf)[i]; break;
        case UINT32: v = (float32)((uint32*)tx->buf)[i]; break;
        case FP16: v = fp16_to_float(((float16 *)tx->buf)[i]); break;
        case FP32: v = ((float32*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "asin_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = asinf(v);
    }
  }else if(out_dtype == FP64){
    float64 *out = (float64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 v;
      switch(tx->dtype){
        case INT64: v = (float64)((int64*)tx->buf)[i]; break;
        case UINT64: v = (float64)((uint64*)tx->buf)[i]; break;
        case FP64: v = ((float64*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "asin_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = asin(v);
    }
  }else if(out_dtype == CMPX64){
    complex64 *in  = (complex64*)tx->buf;
    complex64 *out = (complex64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 a = in[i].real;
      float32 b = in[i].imag;
      float32 z2_r = a*a - b*b;
      float32 z2_i = 2.0f*a*b;
      float32 w_r = 1.0f - z2_r;
      float32 w_i = -z2_i;
      float32 r = sqrtf(w_r*w_r + w_i*w_i);
      float32 s_r = sqrtf((r + w_r) * 0.5f);
      float32 s_i = (w_i >= 0 ? 1.0f : -1.0f) * sqrtf((r - w_r) * 0.5f);
      float32 iz_r = -b;
      float32 iz_i =  a;
      float32 t_r = iz_r + s_r;
      float32 t_i = iz_i + s_i;
      float32 mod = sqrtf(t_r*t_r + t_i*t_i);
      float32 ang = atan2f(t_i, t_r);
      float32 ln_r = logf(mod);
      float32 ln_i = ang;
      out[i].real =  ln_i;
      out[i].imag = -ln_r;
    }
  }else if(out_dtype == CMPX128){
    complex128 *in  = (complex128*)tx->buf;
    complex128 *out = (complex128*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 a = in[i].real;
      float64 b = in[i].imag;
      float64 z2_r = a*a - b*b;
      float64 z2_i = 2.0f*a*b;
      float64 w_r = 1.0f - z2_r;
      float64 w_i = -z2_i;
      float64 r = sqrt(w_r*w_r + w_i*w_i);
      float64 s_r = sqrt((r + w_r) * 0.5f);
      float64 s_i = (w_i >= 0 ? 1.0f : -1.0f) * sqrt((r - w_r) * 0.5f);
      float64 iz_r = -b;
      float64 iz_i =  a;
      float64 t_r = iz_r + s_r;
      float64 t_i = iz_i + s_i;
      float64 mod = sqrt(t_r*t_r + t_i*t_i);
      float64 ang = atan2(t_i, t_r);
      float64 ln_r = log(mod);
      float64 ln_i = ang;
      out[i].real =  ln_i;
      out[i].imag = -ln_r;
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *acos_(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor_t capsule pointer");
    return NULL;
  }
  dtype_t out_dtype;
  switch(tx->dtype){
    case INT8: case UINT8:
    case INT16: case UINT16:
    case INT32: case UINT32: out_dtype = FP32; break;
    case INT64: case UINT64: out_dtype = FP64; break;
    case FP16: out_dtype = FP32; break;
    case FP32: out_dtype = FP32; break;
    case FP64: out_dtype = FP64; break;
    case CMPX64: out_dtype = CMPX64; break;
    case CMPX128: out_dtype = CMPX128; break;
    default: PyErr_SetString(PyExc_TypeError, "acos_() unsupported dtype"); return NULL;
  }
  tensor_t *tz = tensor_empty_like(tx, out_dtype);
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "Failed to allocate output tensor");
    return NULL;
  }
  size_t N = tx->size;
  if(out_dtype == FP32){
    float32 *out = (float32*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 v;
      switch(tx->dtype){
        case INT8: v = (float32)((int8*)tx->buf)[i]; break;
        case UINT8: v = (float32)((uint8*)tx->buf)[i]; break;
        case INT16: v = (float32)((int16*)tx->buf)[i]; break;
        case UINT16: v = (float32)((uint16*)tx->buf)[i]; break;
        case INT32: v = (float32)((int32*)tx->buf)[i]; break;
        case UINT32: v = (float32)((uint32*)tx->buf)[i]; break;
        case FP16: v = fp16_to_float(((float16 *)tx->buf)[i]); break;
        case FP32: v = ((float32*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "acos_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = acosf(v);
    }
  }else if(out_dtype == FP64){
    float64 *out = (float64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 v;
      switch(tx->dtype){
        case INT64: v = (float64)((int64*)tx->buf)[i]; break;
        case UINT64: v = (float64)((uint64*)tx->buf)[i]; break;
        case FP64: v = ((float64*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "acos_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = acos(v);
    }
  }else if(out_dtype == CMPX64){
    complex64 *in  = (complex64*)tx->buf;
    complex64 *out = (complex64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 a = in[i].real;
      float32 b = in[i].imag;
      float32 z2_r = a*a - b*b;
      float32 z2_i = 2.0f*a*b;
      float32 w_r = z2_r - 1.0f;
      float32 w_i = z2_i;
      float32 r = sqrtf(w_r*w_r + w_i*w_i);
      float32 s_r = sqrtf((r + w_r) * 0.5f);
      float32 s_i = (w_i >= 0 ? 1.0f : -1.0f) * sqrtf((r - w_r) * 0.5f);
      float32 t_r = a + s_r;
      float32 t_i = b + s_i;
      float32 mod = sqrtf(t_r*t_r + t_i*t_i);
      float32 ang = atan2f(t_i, t_r);
      float32 ln_r = logf(mod);
      float32 ln_i = ang;
      out[i].real =  ln_i;
      out[i].imag = -ln_r;
    }
  }else if(out_dtype == CMPX128){
    complex128 *in  = (complex128*)tx->buf;
    complex128 *out = (complex128*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 a = in[i].real;
      float64 b = in[i].imag;
      float64 z2_r = a*a - b*b;
      float64 z2_i = 2.0f*a*b;
      float64 w_r = z2_r - 1.0f;
      float64 w_i = z2_i;
      float64 r = sqrt(w_r*w_r + w_i*w_i);
      float64 s_r = sqrt((r + w_r) * 0.5f);
      float64 s_i = (w_i >= 0 ? 1.0f : -1.0f) * sqrt((r - w_r) * 0.5f);
      float64 t_r = a + s_r;
      float64 t_i = b + s_i;
      float64 mod = sqrt(t_r*t_r + t_i*t_i);
      float64 ang = atan2(t_i, t_r);
      float64 ln_r = log(mod);
      float64 ln_i = ang;
      out[i].real =  ln_i;
      out[i].imag = -ln_r;
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *atan_(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor_t capsule pointer");
    return NULL;
  }
  dtype_t out_dtype;
  switch(tx->dtype){
    case INT8: case UINT8:
    case INT16: case UINT16:
    case INT32: case UINT32: out_dtype = FP32; break;
    case INT64: case UINT64: out_dtype = FP64; break;
    case FP16: out_dtype = FP32; break;
    case FP32: out_dtype = FP32; break;
    case FP64: out_dtype = FP64; break;
    case CMPX64: out_dtype = CMPX64; break;
    case CMPX128: out_dtype = CMPX128; break;
    default: PyErr_SetString(PyExc_TypeError, "atan_() unsupported dtype"); return NULL;
  }
  tensor_t *tz = tensor_empty_like(tx, out_dtype);
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "Failed to allocate output tensor");
    return NULL;
  }
  size_t N = tx->size;
  if(out_dtype == FP32){
    float32 *out = (float32*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 v;
      switch(tx->dtype){
        case INT8: v = (float32)((int8*)tx->buf)[i]; break;
        case UINT8: v = (float32)((uint8*)tx->buf)[i]; break;
        case INT16: v = (float32)((int16*)tx->buf)[i]; break;
        case UINT16: v = (float32)((uint16*)tx->buf)[i]; break;
        case INT32: v = (float32)((int32*)tx->buf)[i]; break;
        case UINT32: v = (float32)((uint32*)tx->buf)[i]; break;
        case FP16: v = fp16_to_float(((float16 *)tx->buf)[i]); break;
        case FP32: v = ((float32*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "atan_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = atanf(v);
    }
  }else if(out_dtype == FP64){
    float64 *out = (float64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 v;
      switch(tx->dtype){
        case INT64: v = (float64)((int64*)tx->buf)[i]; break;
        case UINT64: v = (float64)((uint64*)tx->buf)[i]; break;
        case FP64: v = ((float64*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "atan_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = atan(v);
    }
  }else if(out_dtype == CMPX64){
    complex64 *in  = (complex64*)tx->buf;
    complex64 *out = (complex64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 a = in[i].real;
      float32 b = in[i].imag;
      float32 iz_r = -b;
      float32 iz_i =  a;
      float32 n_r = 1.0f + iz_r;
      float32 n_i = iz_i;
      float32 d_r = 1.0f - iz_r;
      float32 d_i = -iz_i;
      float32 denom = d_r*d_r + d_i*d_i;
      float32 r_r = (n_r*d_r + n_i*d_i) / denom;
      float32 r_i = (n_i*d_r - n_r*d_i) / denom;
      float32 mod = sqrtf(r_r*r_r + r_i*r_i);
      float32 ang = atan2f(r_i, r_r);
      float32 ln_r = logf(mod);
      float32 ln_i = ang;
      out[i].real =  0.5f * ln_i;
      out[i].imag = -0.5f * ln_r;
    }
  }else if(out_dtype == CMPX128){
    complex128 *in  = (complex128*)tx->buf;
    complex128 *out = (complex128*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 a = in[i].real;
      float64 b = in[i].imag;
      float64 iz_r = -b;
      float64 iz_i =  a;
      float64 n_r = 1.0 + iz_r;
      float64 n_i = iz_i;
      float64 d_r = 1.0 - iz_r;
      float64 d_i = -iz_i;
      float64 denom = d_r*d_r + d_i*d_i;
      float64 r_r = (n_r*d_r + n_i*d_i) / denom;
      float64 r_i = (n_i*d_r - n_r*d_i) / denom;
      float64 mod = sqrt(r_r*r_r + r_i*r_i);
      float64 ang = atan2(r_i, r_r);
      float64 ln_r = log(mod);
      float64 ln_i = ang;
      out[i].real =  0.5 * ln_i;
      out[i].imag = -0.5 * ln_r;
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *sinh_(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor_t capsule pointer");
    return NULL;
  }
  dtype_t out_dtype;
  switch(tx->dtype){
    case INT8: case UINT8:
    case INT16: case UINT16:
    case INT32: case UINT32: out_dtype = FP32; break;
    case INT64: case UINT64: out_dtype = FP64; break;
    case FP16: out_dtype = FP32; break;
    case FP32: out_dtype = FP32; break;
    case FP64: out_dtype = FP64; break;
    case CMPX64: out_dtype = CMPX64; break;
    case CMPX128: out_dtype = CMPX128; break;
    default: PyErr_SetString(PyExc_TypeError, "sinh_() unsupported dtype"); return NULL;
  }
  tensor_t *tz = tensor_empty_like(tx, out_dtype);
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "Failed to allocate output tensor");
    return NULL;
  }
  size_t N = tx->size;
  if(out_dtype == FP32){
    float32 *out = (float32*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 v;
      switch(tx->dtype){
        case INT8: v = (float32)((int8*)tx->buf)[i]; break;
        case UINT8: v = (float32)((uint8*)tx->buf)[i]; break;
        case INT16: v = (float32)((int16*)tx->buf)[i]; break;
        case UINT16: v = (float32)((uint16*)tx->buf)[i]; break;
        case INT32: v = (float32)((int32*)tx->buf)[i]; break;
        case UINT32: v = (float32)((uint32*)tx->buf)[i]; break;
        case FP16: v = fp16_to_float(((float16 *)tx->buf)[i]); break;
        case FP32: v = ((float32*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "sinh_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = sinf(v);
    }
  }else if(out_dtype == FP64){
    float64 *out = (float64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 v;
      switch(tx->dtype){
        case INT64: v = (float64)((int64*)tx->buf)[i]; break;
        case UINT64: v = (float64)((uint64*)tx->buf)[i]; break;
        case FP64: v = ((float64*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "sinh_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = sinh(v);
    }
  }else if(out_dtype == CMPX64){
    complex64 *in  = (complex64*)tx->buf;
    complex64 *out = (complex64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 a = in[i].real;
      float32 b = in[i].imag;
      out[i].real = sinhf(a) * cosf(b);
      out[i].imag = coshf(a) * sinf(b);
    }
  }else if(out_dtype == CMPX128){
    complex128 *in  = (complex128*)tx->buf;
    complex128 *out = (complex128*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 a = in[i].real;
      float64 b = in[i].imag;
      out[i].real = sinh(a) * cos(b);
      out[i].imag = cosh(a) * sin(b);
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *cosh_(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor_t capsule pointer");
    return NULL;
  }
  dtype_t out_dtype;
  switch(tx->dtype){
    case INT8: case UINT8:
    case INT16: case UINT16:
    case INT32: case UINT32: out_dtype = FP32; break;
    case INT64: case UINT64: out_dtype = FP64; break;
    case FP16: out_dtype = FP32; break;
    case FP32: out_dtype = FP32; break;
    case FP64: out_dtype = FP64; break;
    case CMPX64: out_dtype = CMPX64; break;
    case CMPX128: out_dtype = CMPX128; break;
    default: PyErr_SetString(PyExc_TypeError, "cosh_() unsupported dtype"); return NULL;
  }
  tensor_t *tz = tensor_empty_like(tx, out_dtype);
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "Failed to allocate output tensor");
    return NULL;
  }
  size_t N = tx->size;
  if(out_dtype == FP32){
    float32 *out = (float32*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 v;
      switch(tx->dtype){
        case INT8: v = (float)((int8*)tx->buf)[i]; break;
        case UINT8: v = (float)((uint8*)tx->buf)[i]; break;
        case INT16: v = (float)((int16*)tx->buf)[i]; break;
        case UINT16: v = (float)((uint16*)tx->buf)[i]; break;
        case INT32: v = (float)((int32*)tx->buf)[i]; break;
        case UINT32: v = (float)((uint32*)tx->buf)[i]; break;
        case FP16: v = fp16_to_float(((float16 *)tx->buf)[i]); break;
        case FP32: v = ((float32*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "cosh_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = coshf(v);
    }
  }else if(out_dtype == FP64){
    float64 *out = (float64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 v;
      switch(tx->dtype){
        case INT64: v = (float64)((int64*)tx->buf)[i]; break;
        case UINT64: v = (float64)((uint64*)tx->buf)[i]; break;
        case FP64: v = ((float64*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "cosh_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = cosh(v);
    }
  }else if(CMPX64){
    complex64 *in  = (complex64*)tx->buf;
    complex64 *out = (complex64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 a = in[i].real;
      float32 b = in[i].imag;
      out[i].real = coshf(a) * cosf(b);
      out[i].imag = sinhf(a) * sinf(b);
    }
  }else if(CMPX128){
    complex128 *in  = (complex128*)tx->buf;
    complex128 *out = (complex128*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 a = in[i].real;
      float64 b = in[i].imag;
      out[i].real = cosh(a) * cos(b);
      out[i].imag = sinh(a) * sin(b);
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *tanh_(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor_t capsule pointer");
    return NULL;
  }
  dtype_t out_dtype;
  switch(tx->dtype){
    case INT8: case UINT8:
    case INT16: case UINT16:
    case INT32: case UINT32: out_dtype = FP32; break;
    case INT64: case UINT64: out_dtype = FP64; break;
    case FP16: out_dtype = FP32; break;
    case FP32: out_dtype = FP32; break;
    case FP64: out_dtype = FP64; break;
    case CMPX64: out_dtype = CMPX64; break;
    case CMPX128: out_dtype = CMPX128; break;
    default: PyErr_SetString(PyExc_TypeError, "tanh_() unsupported dtype"); return NULL;
  }
  tensor_t *tz = tensor_empty_like(tx, out_dtype);
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "Failed to allocate output tensor");
    return NULL;
  }
  size_t N = tx->size;
  if(out_dtype == FP32){
    float32 *out = (float32*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 v;
      switch(tx->dtype){
        case INT8: v = (float32)((int8*)tx->buf)[i]; break;
        case UINT8: v = (float32)((uint8*)tx->buf)[i]; break;
        case INT16: v = (float32)((int16*)tx->buf)[i]; break;
        case UINT16: v = (float32)((uint16*)tx->buf)[i]; break;
        case INT32: v = (float32)((int32*)tx->buf)[i]; break;
        case UINT32: v = (float32)((uint32*)tx->buf)[i]; break;
        case FP16: v = fp16_to_float(((float16 *)tx->buf)[i]); break;
        case FP32: v = ((float32*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "tanh_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = tanhf(v);
    }
  }else if(out_dtype == FP64){
    float64 *out = (float64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 v;
      switch(tx->dtype){
        case INT64: v = (float64)((int64*)tx->buf)[i]; break;
        case UINT64: v = (float64)((uint64*)tx->buf)[i]; break;
        case FP64: v = ((float64*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "tanh_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = tanh(v);
    }
  }else if(out_dtype == CMPX64){
    complex64 *in  = (complex64*)tx->buf;
    complex64 *out = (complex64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 a = in[i].real;
      float32 b = in[i].imag;
      float32 denom = coshf(2.0f*a) + cosf(2.0f*b);
      out[i].real = sinhf(2.0f*a) / denom;
      out[i].imag = sinf(2.0f*b)  / denom;
    }
  }else if(out_dtype == CMPX128){
    complex128 *in  = (complex128*)tx->buf;
    complex128 *out = (complex128*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 a = in[i].real;
      float64 b = in[i].imag;
      float64 denom = cosh(2.0*a) + cos(2.0*b);
      out[i].real = sinh(2.0*a) / denom;
      out[i].imag = sin(2.0*b)  / denom;
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *asinh_(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor_t capsule pointer");
    return NULL;
  }
  dtype_t out_dtype;
  switch(tx->dtype){
    case INT8: case UINT8:
    case INT16: case UINT16:
    case INT32: case UINT32: out_dtype = FP32; break;
    case INT64: case UINT64: out_dtype = FP64; break;
    case FP16: out_dtype = FP32; break;
    case FP32: out_dtype = FP32; break;
    case FP64: out_dtype = FP64; break;
    case CMPX64: out_dtype = CMPX64; break;
    case CMPX128: out_dtype = CMPX128; break;
    default: PyErr_SetString(PyExc_TypeError, "asinh_() unsupported dtype"); return NULL;
  }
  tensor_t *tz = tensor_empty_like(tx, out_dtype);
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "Failed to allocate output tensor");
    return NULL;
  }
  size_t N = tx->size;
  if(out_dtype == FP32){
    float32 *out = (float32*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 v;
      switch(tx->dtype){
        case INT8: v = (float32)((int8*)tx->buf)[i]; break;
        case UINT8: v = (float32)((uint8*)tx->buf)[i]; break;
        case INT16: v = (float32)((int16*)tx->buf)[i]; break;
        case UINT16: v = (float32)((uint16*)tx->buf)[i]; break;
        case INT32: v = (float32)((int32*)tx->buf)[i]; break;
        case UINT32: v = (float32)((uint32*)tx->buf)[i]; break;
        case FP16: v = fp16_to_float(((float16 *)tx->buf)[i]); break;
        case FP32: v = ((float32*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "asinh_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = asinf(v);
    }
  }else if(out_dtype == FP64){
    float64 *out = (float64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 v;
      switch(tx->dtype){
        case INT64: v = (float64)((int64*)tx->buf)[i]; break;
        case UINT64: v = (float64)((uint64*)tx->buf)[i]; break;
        case FP64: v = ((float64*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "asinh_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = asinh(v);
    }
  }else if(out_dtype == CMPX64){
    complex64 *in  = (complex64*)tx->buf;
    complex64 *out = (complex64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 a = in[i].real;
      float32 b = in[i].imag;
      float32 z2_r = a*a - b*b;
      float32 z2_i = 2.0f*a*b;
      float32 w_r = z2_r + 1.0f;
      float32 w_i = z2_i;
      float32 r = sqrtf(w_r*w_r + w_i*w_i);
      float32 s_r = sqrtf((r + w_r) * 0.5f);
      float32 s_i = (w_i >= 0 ? 1.0f : -1.0f) * sqrtf((r - w_r) * 0.5f);
      float32 t_r = a + s_r;
      float32 t_i = b + s_i;
      float32 mod = sqrtf(t_r*t_r + t_i*t_i);
      float32 ang = atan2f(t_i, t_r);
      out[i].real = logf(mod);
      out[i].imag = ang;
    }
  }else if(out_dtype == CMPX128){
    complex128 *in  = (complex128*)tx->buf;
    complex128 *out = (complex128*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 a = in[i].real;
      float64 b = in[i].imag;
      float64 z2_r = a*a - b*b;
      float64 z2_i = 2.0*a*b;
      float64 w_r = z2_r + 1.0;
      float64 w_i = z2_i;
      float64 r = sqrt(w_r*w_r + w_i*w_i);
      float64 s_r = sqrt((r + w_r) * 0.5);
      float64 s_i = (w_i >= 0 ? 1.0 : -1.0) * sqrt((r - w_r) * 0.5);
      float64 t_r = a + s_r;
      float64 t_i = b + s_i;
      float64 mod = sqrt(t_r*t_r + t_i*t_i);
      float64 ang = atan2(t_i, t_r);
      out[i].real = log(mod);
      out[i].imag = ang;
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *acosh_(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor_t capsule pointer");
    return NULL;
  }
  dtype_t out_dtype;
  switch(tx->dtype){
    case INT8: case UINT8:
    case INT16: case UINT16:
    case INT32: case UINT32: out_dtype = FP32; break;
    case INT64: case UINT64: out_dtype = FP64; break;
    case FP16: out_dtype = FP32; break;
    case FP32: out_dtype = FP32; break;
    case FP64: out_dtype = FP64; break;
    case CMPX64: out_dtype = CMPX64; break;
    case CMPX128: out_dtype = CMPX128; break;
    default: PyErr_SetString(PyExc_TypeError, "acosh_() unsupported dtype"); return NULL;
  }
  tensor_t *tz = tensor_empty_like(tx, out_dtype);
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "Failed to allocate output tensor");
    return NULL;
  }
  size_t N = tx->size;
  if(out_dtype == FP32){
    float32 *out = (float32*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 v;
      switch(tx->dtype){
        case INT8: v = (float32)((int8*)tx->buf)[i]; break;
        case UINT8: v = (float32)((uint8*)tx->buf)[i]; break;
        case INT16: v = (float32)((int16*)tx->buf)[i]; break;
        case UINT16: v = (float32)((uint16*)tx->buf)[i]; break;
        case INT32: v = (float32)((int32*)tx->buf)[i]; break;
        case UINT32: v = (float32)((uint32*)tx->buf)[i]; break;
        case FP16: v = fp16_to_float(((float16 *)tx->buf)[i]); break;
        case FP32: v = ((float32*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "acosh_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = acoshf(v);
    }
  }else if(out_dtype == FP64){
    float64 *out = (float64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 v;
      switch(tx->dtype){
        case INT64: v = (float64)((int64*)tx->buf)[i]; break;
        case UINT64: v = (float64)((uint64*)tx->buf)[i]; break;
        case FP64: v = ((float64*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "acosh_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = acosh(v);
    }
  }else if(out_dtype == CMPX64){
    complex64 *in  = (complex64*)tx->buf;
    complex64 *out = (complex64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 a = in[i].real;
      float32 b = in[i].imag;
      float32 zm1_r = a - 1.0f;
      float32 zm1_i = b;
      float32 zp1_r = a + 1.0f;
      float32 zp1_i = b;
      float32 r1 = sqrtf(zm1_r*zm1_r + zm1_i*zm1_i);
      float32 s1_r = sqrtf((r1 + zm1_r) * 0.5f);
      float32 s1_i = (zm1_i >= 0 ? 1.0f : -1.0f) * sqrtf((r1 - zm1_r) * 0.5f);
      float32 r2 = sqrtf(zp1_r*zp1_r + zp1_i*zp1_i);
      float32 s2_r = sqrtf((r2 + zp1_r) * 0.5f);
      float32 s2_i = (zp1_i >= 0 ? 1.0f : -1.0f) * sqrtf((r2 - zp1_r) * 0.5f);
      float32 prod_r = s1_r*s2_r - s1_i*s2_i;
      float32 prod_i = s1_r*s2_i + s1_i*s2_r;
      float32 t_r = a + prod_r;
      float32 t_i = b + prod_i;
      float32 mod = sqrtf(t_r*t_r + t_i*t_i);
      float32 ang = atan2f(t_i, t_r);
      out[i].real = logf(mod);
      out[i].imag = ang;
    }
  }else if(out_dtype == CMPX128){
    complex128 *in  = (complex128*)tx->buf;
    complex128 *out = (complex128*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 a = in[i].real;
      float64 b = in[i].imag;
      float64 zm1_r = a - 1.0;
      float64 zm1_i = b;
      float64 zp1_r = a + 1.0;
      float64 zp1_i = b;
      float64 r1 = sqrt(zm1_r*zm1_r + zm1_i*zm1_i);
      float64 s1_r = sqrt((r1 + zm1_r) * 0.5);
      float64 s1_i = (zm1_i >= 0 ? 1.0 : -1.0) * sqrt((r1 - zm1_r) * 0.5);
      float64 r2 = sqrt(zp1_r*zp1_r + zp1_i*zp1_i);
      float64 s2_r = sqrt((r2 + zp1_r) * 0.5);
      float64 s2_i = (zp1_i >= 0 ? 1.0 : -1.0) * sqrt((r2 - zp1_r) * 0.5);
      float64 prod_r = s1_r*s2_r - s1_i*s2_i;
      float64 prod_i = s1_r*s2_i + s1_i*s2_r;
      float64 t_r = a + prod_r;
      float64 t_i = b + prod_i;
      float64 mod = sqrt(t_r*t_r + t_i*t_i);
      float64 ang = atan2(t_i, t_r);
      out[i].real = log(mod);
      out[i].imag = ang;
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *atanh_(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor_t capsule pointer");
    return NULL;
  }
  dtype_t out_dtype;
  switch(tx->dtype){
    case INT8: case UINT8:
    case INT16: case UINT16:
    case INT32: case UINT32: out_dtype = FP32; break;
    case INT64: case UINT64: out_dtype = FP64; break;
    case FP16: out_dtype = FP32; break;
    case FP32: out_dtype = FP32; break;
    case FP64: out_dtype = FP64; break;
    case CMPX64: out_dtype = CMPX64; break;
    case CMPX128: out_dtype = CMPX128; break;
    default: PyErr_SetString(PyExc_TypeError, "atanh_() unsupported dtype"); return NULL;
  }
  tensor_t *tz = tensor_empty_like(tx, out_dtype);
  if(!tz){
    PyErr_SetString(PyExc_RuntimeError, "Failed to allocate output tensor");
    return NULL;
  }
  size_t N = tx->size;
  if(out_dtype == FP32){
    float32 *out = (float32*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 v;
      switch(tx->dtype){
        case INT8: v = (float32)((int8*)tx->buf)[i]; break;
        case UINT8: v = (float32)((uint8*)tx->buf)[i]; break;
        case INT16: v = (float32)((int16*)tx->buf)[i]; break;
        case UINT16: v = (float32)((uint16*)tx->buf)[i]; break;
        case INT32: v = (float32)((int32*)tx->buf)[i]; break;
        case UINT32: v = (float32)((uint32*)tx->buf)[i]; break;
        case FP16: v = fp16_to_float(((float16 *)tx->buf)[i]); break;
        case FP32: v = ((float32*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "atanh_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = atanhf(v);
    }
  }else if(out_dtype == FP64){
    float64 *out = (float64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 v;
      switch(tx->dtype){
        case INT64: v = (float64)((int64*)tx->buf)[i]; break;
        case UINT64: v = (float64)((uint64*)tx->buf)[i]; break;
        case FP64: v = ((float64*)tx->buf)[i]; break;
        default:
          PyErr_SetString(PyExc_TypeError, "atanh_(): internal unsupported dtype");
          destroy(tz);
          return NULL;
      }
      out[i] = atanh(v);
    }
  }else if(out_dtype == CMPX64){
    complex64 *in  = (complex64*)tx->buf;
    complex64 *out = (complex64*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float32 a = in[i].real;
      float32 b = in[i].imag;
      float32 n_r = 1.0f + a;
      float32 n_i = b;
      float32 d_r = 1.0f - a;
      float32 d_i = -b;
      float32 denom = d_r*d_r + d_i*d_i;
      float32 r_r = (n_r*d_r + n_i*d_i) / denom;
      float32 r_i = (n_i*d_r - n_r*d_i) / denom;
      float32 mod = sqrtf(r_r*r_r + r_i*r_i);
      float32 ang = atan2f(r_i, r_r);
      out[i].real = 0.5f * logf(mod);
      out[i].imag = 0.5f * ang;
    }
  }else if(out_dtype == CMPX128){
    complex128 *in  = (complex128*)tx->buf;
    complex128 *out = (complex128*)tz->buf;
    for(size_t i = 0; i < N; i++){
      float64 a = in[i].real;
      float64 b = in[i].imag;
      float64 n_r = 1.0 + a;
      float64 n_i = b;
      float64 d_r = 1.0 - a;
      float64 d_i = -b;
      float64 denom = d_r*d_r + d_i*d_i;
      float64 r_r = (n_r*d_r + n_i*d_i) / denom;
      float64 r_i = (n_i*d_r - n_r*d_i) / denom;
      float64 mod = sqrt(r_r*r_r + r_i*r_i);
      float64 ang = atan2(r_i, r_r);
      out[i].real = 0.5 * log(mod);
      out[i].imag = 0.5 * ang;
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *sgn_(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor_t capsule pointer");
    return NULL;
  }
  tensor_t *tz = tensor_empty_like(tx, tx->dtype);
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Memory allocation failed");
    return NULL;
  }
  size_t N = tx->size;
  switch(tx->dtype){
    case INT8: {
      int8 *in = (int8 *)tx->buf;
      int8 *out = (int8 *)tz->buf;
      for(size_t i = 0; i < N; i++){ out[i] = (in[i] > 0) - (in[i] < 0); }
      break;
    }
    case INT16: {
      int16 *in = (int16 *)tx->buf;
      int16 *out = (int16 *)tz->buf;
      for(size_t i = 0; i < N; i++){ out[i] = (in[i] > 0) - (in[i] < 0); }
      break;
    }
    case INT32: {
      int32 *in = (int32 *)tx->buf;
      int32 *out = (int32 *)tz->buf;
      for(size_t i = 0; i < N; i++){ out[i] = (in[i] > 0) - (in[i] < 0); }
      break;
    }
    case INT64: {
      int64 *in = (int64 *)tx->buf;
      int64 *out = (int64 *)tx-> buf;
      for(size_t i = 0; i < N; i++){ out[i] = (in[i] > 0) - (in[i] < 0); }
      break;
    }
    case FP16: {
      float16 *in = (float16 *)tx->buf;
      float16 *out = (float16 *)tz->buf;
      for(size_t i = 0; i < N; i++){
        float32 x = fp16_to_float(in[i]);
        out[i] = float_to_fp16((x > 0.f) - (x < 0.f));
      }
      break;
    }
    case FP32: {
      float32 *in = (float32 *)tx->buf;
      float32 *out = (float32 *)tz->buf;
      for(size_t i = 0; i < N; i++){ out[i] = (in[i] > 0.f) - (in[i] < 0.f); }
      break;
    }
    case FP64: {
      float64 *in = (float64 *)tx->buf;
      float64 *out = (float64 *)tz->buf;
      for(size_t i = 0; i < N; i++){ out[i] = (in[i] > 0) - (in[i] < 0); }
      break;
    }
    case CMPX64: {
      complex64 *in = (complex64 *)tx->buf;
      complex64 *out = (complex64 *)tz->buf;
      for(size_t i = 0; i < N; i++){
        float32 mag = sqrtf(in[i].real * in[i].real + in[i].imag * in[i].imag);
        if(mag == 0.0f){
          out[i].real = 0.0f;
          out[i].imag = 0.0f;
        }else{
          out[i].real = in[i].real / mag;
          out[i].imag = in[i].imag / mag;
        }
      }
      break;
    }
    case CMPX128: {
      complex128 *in = (complex128 *)tx->buf;
      complex128 *out = (complex128 *)tz->buf;
      for(size_t i = 0; i < N; i++){
        float64 mag = sqrt(in[i].real * in[i].real + in[i].imag * in[i].imag);
        if(mag == 0.0){
          out[i].real = 0.0;
          out[i].imag = 0.0;
        }else{
          out[i].real = in[i].real / mag;
          out[i].imag = in[i].imag / mag;
        }
      }
      break;
    }
    case UINT8: {
      uint8 *in = (uint8 *)tx->buf;
      uint8 *out = (uint8 *)tz->buf;
      for(size_t i = 0; i < N; i++){ out[i] = (in[i] == 0) ? 0 : 1; }
      break;
    }
    case UINT16: {
      uint16 *in = (uint16 *)tx->buf;
      uint16 *out = (uint16 *)tz->buf;
      for(size_t i = 0; i < N; i++){ out[i] = (in[i] == 0) ? 0 : 1; }
      break;
    }
    case UINT32: {
      uint32 *in = (uint32 *)tx->buf;
      uint32 *out = (uint32 *)tz->buf;
      for(size_t i = 0; i < N; i++){ out[i] = (in[i] == 0) ? 0 : 1; }
      break;
    }
    case UINT64: {
      uint64 *in = (uint64 *)tx->buf;
      uint64 *out = (uint64 *)tz->buf;
      for(size_t i = 0; i < N; i++){ out[i] = (in[i] == 0) ? 0 : 1; }
      break;
    }
  }
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *eye_(PyObject *self, PyObject *args){
  size_t m, n;
  const char *fmt;
  if(!PyArg_ParseTuple(args, "nns", &m, &n, &fmt)) return NULL;
  dtype_t dtype = getdtype(*fmt);
  if(dtype == ERROR){
    PyErr_Format(PyExc_ValueError, "Invalid fmt string %c", *fmt);
    return NULL;
  }
  if(dtype == CMPX64 || dtype == CMPX128){
    PyErr_SetString(PyExc_RuntimeError, "eye_() is not supported for complex dtype");
    return NULL;
  }
  tensor_t *t = tt_malloc(sizeof(tensor_t));
  if(!t) return PyErr_NoMemory();
  t->dtype = dtype;
  t->element_size = getsize(dtype);
  t->device = (device_t){CPU, 0};
  t->ndim = 2;
  t->size = m * n;
  t->shape = tt_malloc(2 * sizeof(size_t));
  if(!t->shape){
    free(t);
    return PyErr_NoMemory();
  }
  t->shape[0] = m;
  t->shape[1] = n;
  t->stride = tt_malloc(2 * sizeof(size_t));
  if(!t->stride){
    free(t->shape);
    free(t);
    return PyErr_NoMemory();
  }
  t->stride[1] = 1;
  t->stride[0] = n;
  storage_t *storage = tt_malloc(sizeof(storage_t));
  if(!storage){
    destroy(t);
    return PyErr_NoMemory();
  }
  storage->bytes = t->size * t->element_size;
  storage->device = t->device;
  storage->refcount = 1;
  storage->ptr = tt_malloc(storage->bytes);
  if(!storage->ptr){
    free(storage);
    destroy(t);
    return PyErr_NoMemory();
  }
  memset(storage->ptr, 0, storage->bytes);
  t->storage = storage;
  t->buf = storage->ptr;
  size_t diag = (m < n) ? m : n;
  for(size_t i = 0; i < diag; i++){
    size_t idx = i * t->stride[0] + i * t->stride[1];
    switch(dtype){
      case BOOL: ((bool*)t->buf)[idx] = 1; break;
      case INT8: ((int8*)t->buf)[idx] = 1; break;
      case UINT8: ((uint8*)t->buf)[idx] = 1; break;
      case INT16: ((int16*)t->buf)[idx] = 1; break;
      case UINT16: ((uint16*)t->buf)[idx] = 1; break;
      case INT32: ((int32*)t->buf)[idx] = 1; break;
      case UINT32: ((uint32*)t->buf)[idx] = 1; break;
      case INT64: ((int64*)t->buf)[idx] = 1; break;
      case UINT64: ((uint64*)t->buf)[idx] = 1; break;
      case FP16: ((float16*)t->buf)[idx] = float_to_fp16(1.0f); break;
      case FP32: ((float32*)t->buf)[idx] = 1.0f; break;
      case FP64: ((float64*)t->buf)[idx] = 1.0; break;
      default:
        destroy(t);
        PyErr_SetString(PyExc_TypeError, "Unsupported dtype for eye");
        return NULL;
    }
  }
  return PyCapsule_New(t, "tensor_t on CPU", capsule_destroyer);
}

static PyMethodDef methods[] = {
  {"add", add, METH_VARARGS, "element-wise 'add' operation on tensor"},
  {"sub", sub, METH_VARARGS, "element-wise 'sub' operation tensor"},
  {"mul", mul, METH_VARARGS, "element-wise 'mul' operation on tensor"},
  {"tdiv", tdiv, METH_VARARGS, "element-wise 'tdiv' operation tensor"},
  {"fdiv", fdiv_, METH_VARARGS, "element-wise 'fdiv' operation on tensor"},
  {"pow", pow_, METH_VARARGS, "element-wise 'pow' operation on tensor"},
  {"mod", mod_, METH_VARARGS, "element-wise 'mod' operation on tensor"},
  {"sqrt", sqrt_, METH_VARARGS, "element-wise 'sqrt' operation on tensor"},
  {"cbrt", cbrt_, METH_VARARGS, "element-wise 'cbrt' operation on tensor"},
  {"floor", floor_, METH_VARARGS, "element-wise 'floor' operation on tensor"},
  {"ceil", ceil_, METH_VARARGS, "element-wise 'ceil' operation on tensor"},
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
  {"bitwise_and", bitwise_and, METH_VARARGS, "element-wise bitwise 'and' operation on tensor"},
  {"bitwise_nand", bitwise_nand, METH_VARARGS, "element-wise bitwise 'nand' operation on tensor"},
  {"bitwise_or", bitwise_or, METH_VARARGS, "element-wise bitwise 'or' operation on tensor"},
  {"bitwise_nor", bitwise_nor, METH_VARARGS, "element-wise bitwise 'nor' operation on tensor"},
  {"bitwise_not", bitwise_not, METH_VARARGS, "element-wise bitwise 'not' operation on tensor"},
  {"bitwise_xor", bitwise_xor, METH_VARARGS, "element-wise bitwise 'xor' operation on tensor"},
  {"bitwise_xnor", bitwise_xnor, METH_VARARGS, "element-wise bitwise 'xnor' operation on tensor"},
  {"logical_and", logical_and, METH_VARARGS, "element-wise logical 'and' operation on tensor"},
  {"logical_nand", logical_nand, METH_VARARGS, "element-wise logical 'nand' operation on tensor"},
  {"logical_or", logical_or, METH_VARARGS, "element-wise logical 'or' operation on tensor"},
  {"logical_nor", logical_nor, METH_VARARGS, "element-wise logical 'nor' operation on tensor"},
  {"logical_not", logical_not, METH_VARARGS, "element-wise logical 'not' operation on tensor"},
  {"logical_xor", logical_xor, METH_VARARGS, "element-wise logical 'xor' operation on tensor"},
  {"logical_xnor", logical_xnor, METH_VARARGS, "element-wise logical 'xnor' operation on tensor"},
  {"sum", (PyCFunction)sum, METH_VARARGS | METH_KEYWORDS, "returns the sum of the tensor"},
  {"bmm", bmm, METH_VARARGS, "compute batch matrix multiplication on tensor"},
  {"conj", conj_, METH_VARARGS, "conjugate of of complex tensor"},
  {"real", real, METH_VARARGS, "real values from complex tensor"},
  {"imag", imag, METH_VARARGS, "imag values from complex tensor"},
  {"exp", exp_, METH_VARARGS, "computes exponential of tensor"},
  {"log", log_, METH_VARARGS, "computes log of tensor to the base `e`"},
  {"log2", log2_, METH_VARARGS, "computes log of tensor to the base `2`"},
  {"log10", log10_, METH_VARARGS, "computes log of tensor to the base `10`"},
  {"sin", sin_, METH_VARARGS, "computes sine of tensor"},
  {"cos", cos_, METH_VARARGS, "computes cosine of tensor"},
  {"tan", tan_, METH_VARARGS, "computes tangent of tensor"},
  {"asin", asin_, METH_VARARGS, "computes arc sine of tensor"},
  {"acos", acos_, METH_VARARGS, "computes arc cosine of tensor"},
  {"atan", atan_, METH_VARARGS, "computes arc cosine of tensor"},
  {"sinh", sinh_, METH_VARARGS, "computes hyperbolic sine of tensor"},
  {"cosh", cosh_, METH_VARARGS, "computes hyperbolic cosine of tensor"},
  {"tanh", tanh_, METH_VARARGS, "computes hyperbolic tangent of tensor"},
  {"asinh", asinh_, METH_VARARGS, "computes arc hyperbolic sine of tensor"},
  {"acosh", acosh_, METH_VARARGS, "computes arc hyperbolic cosine of tensor"},
  {"atanh", atanh_, METH_VARARGS, "computes are hyperbolic tangent of tensor"},
  {"sgn", sgn_, METH_VARARGS, "computes signum function on tensor"},
  {"eye", eye_, METH_VARARGS, "returns identity tensor"},
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
