#include <python3.10/Python.h>
#include <python3.10/longobject.h>
#include <python3.10/pyerrors.h>
#include "../tensor.h"

void capsule_destroyer(PyObject *capsule){
  tensor_t *t = PyCapsule_GetPointer(capsule, "tensor_t on CPU");
  if(t){
    destroy(t);
  }
}

static inline long long py_to_int(PyObject *o){ return PyLong_AsLongLong(o); }
static inline unsigned long long py_to_uint(PyObject *o){ return PyLong_AsUnsignedLongLongMask(o); }
static inline double py_to_float(PyObject *o){ return PyFloat_AsDouble(o); }

dtype_t get_dtype(const char fmt){
  switch(fmt){
    case '?': return BOOL;
    case 'b': return INT8;
    case 'B': return UINT8;
    case 'h': return INT16;
    case 'H': return UINT16;
    case 'i': return INT32;
    case 'I': return UINT32;
    case 'l': return INT64;
    case 'L': return UINT64;
    case 'e': return FP16;
    case 'f': return FP32;
    case 'd': return FP64;
    case 'F': return CMPX64;
    case 'D': return CMPX128;
    default: return ERROR;
  }
}

static PyObject *__list__(PyObject *list, PyObject *shape, Py_ssize_t length, const char fmt){
  dtype_t dtype = get_dtype(fmt);
  if(dtype == ERROR){
    PyErr_Format(PyExc_TypeError, "Invalid dtype fmt: '%c'", fmt);
    return NULL;
  }
  tensor_t *t = malloc(sizeof(tensor_t));
  if(!t){
    PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
    return NULL;
  }
  t->dtype = dtype;
  t->size = length;
  t->ndim = PyTuple_Size(shape);
  t->shape = malloc(sizeof(size_t) * t->ndim);
  if(!t->shape){
    free(t);
    PyErr_SetString(PyExc_RuntimeError, "tensor_t.shape allocation failed!");
    return NULL;
  }
  size_t expected = 1;
  for(size_t i = 0; i < t->ndim; i++){
    PyObject *dim = PyTuple_GetItem(shape, i);
    if(!PyLong_Check(dim)){
      PyErr_SetString(PyExc_TypeError, "shape must contain integers");
      free(t->shape);
      free(t);
      return NULL;
    }
    t->shape[i] = PyLong_AsSize_t(dim);
    expected *= t->shape[i];
  }
  if(expected != (size_t)length){
    PyErr_SetString(PyExc_ValueError, "shape does not match number of elements");
    free(t->shape);
    free(t);
    return NULL;
  }
  t->stride = malloc(sizeof(size_t) * t->ndim);
  if(!t->stride){
    destroy(t);
    PyErr_SetString(PyExc_RuntimeError, "tensor_t.stride allocation failed!");
    return NULL;
  }
  t->stride[t->ndim-1] = 1;
  for(int i = (int)t->ndim - 2; i >= 0; i--){ t->stride[i] = t->shape[i + 1] * t->stride[i+1]; }
  t->element_size = getsize(dtype);
  t->device = (device_t){CPU, 0};
  t->storage = malloc(sizeof(storage_t));
  if(!t->storage){
    free(t->shape);
    free(t);
    PyErr_SetString(PyExc_RuntimeError, "tensor_t.storage allocation failed!");
    return NULL;
  }
  t->storage->bytes = t->size * t->element_size;
  t->storage->device = t->device;
  t->storage->refcount = 1;
  t->storage->ptr = malloc(t->storage->bytes);
  if(!t->storage->ptr){
    free(t->shape);
    free(t->storage);
    free(t);
    PyErr_SetString(PyExc_RuntimeError, "tensor_t.storage.ptr allocation failed!");
    return NULL;
  }
  t->buf = t->storage->ptr;
  void *dest = t->storage->ptr;
  for(Py_ssize_t i = 0; i < length; i++){
    PyObject *item = PyList_GetItem(list, i);
    char *p = (char *)dest + i * t->element_size;
    switch(t->dtype){
      case BOOL: *(bool *)p = PyObject_IsTrue(item); break;
      case INT8: *(int8 *)p = (int8)py_to_int(item); break;
      case UINT8: *(uint8 *)p = (uint8)py_to_uint(item); break;
      case INT16: *(int16 *)p = (int16)py_to_int(item); break;
      case UINT16: *(uint16 *)p = (uint16)py_to_uint(item); break;
      case INT32: *(int32 *)p = (int32)py_to_int(item); break;
      case UINT32: *(uint32 *)p = (uint32)py_to_uint(item); break;
      case INT64: *(int64 *)p = (int64)py_to_int(item); break;
      case UINT64: *(uint64 *)p = (uint64)py_to_uint(item); break;
      case FP16: {
        float v = (float)py_to_float(item);
        *(float16 *)p = float_to_fp16(v);
        break;
      }
      case FP32: *(float32 *)p = (float32)py_to_float(item); break;
      case FP64: *(float64 *)p = (float64)py_to_float(item); break;
      case CMPX64: {
        Py_complex c = PyComplex_AsCComplex(item);
        if(PyErr_Occurred()){
          destroy(t);
          return NULL;
        }
        ((complex64 *)p)->real = (float32)c.real;
        ((complex64 *)p)->imag = (float32)c.imag;
        break;
      }
      case CMPX128: {
        Py_complex c = PyComplex_AsCComplex(item);
        if(PyErr_Occurred()){
          destroy(t);
          return NULL;
        }
        ((complex128 *)p)->real = (float64)c.real;
        ((complex128 *)p)->imag = (float64)c.imag;
        break;
      }
      case ERROR: {
        destroy(t);
        PyErr_Format(PyExc_TypeError, "Invalid dtype fmt: '%c'", fmt);
        return NULL;
      }
    }
  }
  return PyCapsule_New(t, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *__scalar__(PyObject *scalar, const char fmt){
  dtype_t dtype = get_dtype(fmt);
  if(dtype == ERROR){
    PyErr_Format(PyExc_TypeError, "Invalid dtype fmt: '%c'", fmt);
    return NULL;
  }
  tensor_t *t = malloc(sizeof(tensor_t));
  if(!t){
    PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
    return NULL;
  }
  if(!t){ PyErr_NoMemory(); return NULL; }
  t->dtype = dtype;
  t->size = 1;
  t->ndim = 0;
  t->shape = NULL;
  t->stride = NULL; // not implemented yet
  t->element_size = getsize(dtype);
  t->device = (device_t){CPU, 0};
  t->storage = malloc(sizeof(storage_t));
  if(!t->storage){
    free(t->storage);
    PyErr_SetString(PyExc_RuntimeError, "tensor_t.storage allocation failed!");
    return NULL;
  }
  t->storage->bytes = t->element_size;
  t->storage->device = t->device;
  t->storage->refcount = 1;
  t->storage->ptr = malloc(t->storage->bytes);
  if(!t->storage->ptr){
    free(t->storage->ptr);
    PyErr_SetString(PyExc_RuntimeError, "tensor_t.storage.ptr allocation failed!");
    return NULL;
  }
  t->buf = t->storage->ptr;
  void *p = t->buf;
  switch(t->dtype){
    case BOOL: *(bool *)p = PyObject_IsTrue(scalar); break;
    case INT8: *(int8 *)p = (int8)PyLong_AsLongLong(scalar); break;
    case UINT8: *(uint8 *)p = (uint8)PyLong_AsUnsignedLongLongMask(scalar); break;
    case INT16: *(int16 *)p = (int16)PyLong_AsLongLong(scalar); break;
    case UINT16: *(uint16 *)p = (uint16)PyLong_AsUnsignedLongLongMask(scalar); break;
    case INT32: *(int32 *)p = (int32)PyLong_AsLongLong(scalar); break;
    case UINT32: *(uint32 *)p = (uint32)PyLong_AsUnsignedLongLongMask(scalar); break;
    case INT64: *(int64 *)p = (int64)PyLong_AsLongLong(scalar); break;
    case UINT64: *(uint64 *)p = (uint64)PyLong_AsUnsignedLongLongMask(scalar); break;
    case FP16: *(float16 *)p = float_to_fp16((float)PyFloat_AsDouble(scalar)); break;
    case FP32: *(float32 *)p = (float32)PyFloat_AsDouble(scalar); break;
    case FP64: *(float64 *)p = (float64)PyFloat_AsDouble(scalar); break;
    case CMPX64: {
      Py_complex c = PyComplex_AsCComplex(scalar);
      if(PyErr_Occurred()){ destroy(t); return NULL; }
      ((complex64 *)p)->real = (float32)c.real;
      ((complex64 *)p)->imag = (float32)c.imag;
      break;
    }
    case CMPX128: {
      Py_complex c = PyComplex_AsCComplex(scalar);
      if(PyErr_Occurred()){ destroy(t); return NULL; }
      ((complex128 *)p)->real = (float64)c.real;
      ((complex128 *)p)->imag = (float64)c.imag;
      break;
    }
    case ERROR: {
      destroy(t);
      PyErr_Format(PyExc_TypeError, "Invalid dtype fmt: '%c'", fmt);
      return NULL;
    }
  }
  return PyCapsule_New(t, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *tocpu(PyObject *self, PyObject *args){
  PyObject *pyobj;
  PyObject *shape; // empty if scalar
  const char *fmt;
  if(!PyArg_ParseTuple(args, "OOs", &pyobj, &shape, &fmt)) return NULL;
  if(PyTuple_Size(shape) != 0){
    Py_ssize_t length = PyList_Size(pyobj);
    return __list__(pyobj, shape, length, *fmt);
  }else return __scalar__(pyobj, *fmt);
}

static PyObject *topyobj(PyObject *self, PyObject *args){
  PyObject *capsule;
  if(!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
  tensor_t *t = PyCapsule_GetPointer(capsule, "tensor_t on CPU");
  if(!t){
    PyErr_SetString(PyExc_TypeError, "Invalid tensor capsule");
    return NULL;
  }
  if(t->size == 1 && !t->shape){
    char *p = (char *)t->buf;
    switch(t->dtype){
      case BOOL: return PyBool_FromLong(*(bool *)p);
      case INT8: return PyLong_FromLong(*(int8 *)p);
      case UINT8: return PyLong_FromUnsignedLong(*(uint8 *)p);
      case INT16: return PyLong_FromLong(*(int16 *)p);
      case UINT16: return PyLong_FromUnsignedLong(*(uint16 *)p);
      case INT32: return PyLong_FromLong(*(int32 *)p);
      case UINT32: return PyLong_FromUnsignedLong(*(uint32 *)p);
      case INT64: return PyLong_FromLongLong(*(int64 *)p);
      case UINT64: return PyLong_FromUnsignedLongLong(*(uint64 *)p);
      case FP16: {
        float16 h = *(float16 *)p;
        return PyFloat_FromDouble(fp16_to_float(h));
      }
      case FP32: return PyFloat_FromDouble(*(float32 *)p);
      case FP64: return PyFloat_FromDouble(*(float64 *)p);
      case CMPX64: {
        complex64 *c = (complex64 *)p;
        return PyComplex_FromDoubles(c->real, c->imag);
      }
      case CMPX128: {
        complex128 *c = (complex128 *)p;
        return PyComplex_FromDoubles(c->real, c->imag);
      }
      default:
        PyErr_SetString(PyExc_TypeError, "Unsupported dtype");
        return NULL;
    }
  }
  PyObject *list = PyList_New(t->size);
  if(!list) return NULL;
  for(Py_ssize_t i = 0; i < t->size; i++){
    char *p = (char *)t->buf + i * t->element_size;
    PyObject *item = NULL;
    switch(t->dtype){
      case BOOL: item = PyBool_FromLong(*(bool *)p); break;
      case INT8: item = PyLong_FromLong(*(int8 *)p); break;
      case UINT8: item = PyLong_FromUnsignedLong(*(uint8 *)p); break;
      case INT16: item = PyLong_FromLong(*(int16 *)p); break;
      case UINT16: item = PyLong_FromUnsignedLong(*(uint16 *)p); break;
      case INT32: item = PyLong_FromLong(*(int32 *)p); break;
      case UINT32: item = PyLong_FromUnsignedLong(*(uint32 *)p); break;
      case INT64: item = PyLong_FromLongLong(*(int64 *)p); break;
      case UINT64: item = PyLong_FromUnsignedLongLong(*(uint64 *)p); break;
      case FP16: {
        float16_t h = *(float16_t *)p;
        item = PyFloat_FromDouble(fp16_to_float(h));
        break;
      }
      case FP32: item = PyFloat_FromDouble(*(float32 *)p); break;
      case FP64: item = PyFloat_FromDouble(*(float64 *)p); break;
      case CMPX64: {
        complex64 *c = (complex64 *)p;
        item = PyComplex_FromDoubles(c->real, c->imag);
        break;
      }
      case CMPX128: {
        complex128 *c = (complex128 *)p;
        item = PyComplex_FromDoubles(c->real, c->imag);
        break;
      }
      default:
        Py_DECREF(list);
        PyErr_SetString(PyExc_TypeError, "Unsupported dtype");
        return NULL;
    }
    PyList_SET_ITEM(list, i, item);  /* steals ref */
  }
  return list;
}

static PyObject *ndim(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  tensor_t *t = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!t){
    PyErr_SetString(PyExc_RuntimeError, "invalid tensor capsule");
    return NULL;
  }
  return PyLong_FromLong(t->ndim);
}

static PyObject *shape(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_TypeError, "expected tensor capsule");
    return NULL;
  }
  tensor_t *t = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!t){
    PyErr_SetString(PyExc_RuntimeError, "invalid tensor capsule");
    return NULL;
  }
  PyObject *shape_tuple = PyTuple_New(t->ndim);
  if(!shape_tuple){
    PyErr_SetString(PyExc_RuntimeError, "tuple allocation failed");
    return NULL;
  }
  for(int i = 0; i < t->ndim; i++){
    PyTuple_SetItem(shape_tuple, i, PyLong_FromSize_t(t->shape[i]));
  }
  return shape_tuple;
}

static PyObject *stride(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_TypeError, "expected tensor capsule");
    return NULL;
  }
  tensor_t *t = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!t){
    PyErr_SetString(PyExc_RuntimeError, "invalid tensor capsule");
    return NULL;
  }
  PyObject *stride_tuple = PyTuple_New(t->ndim);
  if(!stride_tuple){
    PyErr_SetString(PyExc_RuntimeError, "tuple allocation failed");
    return NULL;
  }
  for(int i = 0; i < t->ndim; i++){
    PyTuple_SetItem(stride_tuple, i, PyLong_FromSize_t(t->stride[i]));
  }
  return stride_tuple;
}

static PyObject *device(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_TypeError, "expected tensor capsule");
    return NULL;
  }
  tensor_t *t = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!t){
    PyErr_SetString(PyExc_RuntimeError, "invalid tensor capsule");
    return NULL;
  }
  const char *type = (t->device.type == CPU) ? "CPU" : "CUDA";
  return Py_BuildValue("(si)", type, t->device.index);
}

static PyObject *dtype(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_TypeError, "expected tensor capsule");
    return NULL;
  }
  tensor_t *t = PyCapsule_GetPointer(x, "tensor_t on CPU");
  if(!t){
    PyErr_SetString(PyExc_RuntimeError, "invalid tensor capsule");
    return NULL;
  }
  switch(t->dtype){
    case BOOL: return PyUnicode_FromString("bool");
    case INT8: return PyUnicode_FromString("char");
    case UINT8: return PyUnicode_FromString("unsigned char");
    case INT16: return PyUnicode_FromString("short");
    case UINT16: return PyUnicode_FromString("unsigned short");
    case INT32: return PyUnicode_FromString("int");
    case UINT32: return PyUnicode_FromString("unsigned int");
    case INT64: return PyUnicode_FromString("long");
    case UINT64: return PyUnicode_FromString("unsigned long");
    case FP16: return PyUnicode_FromString("half");
    case FP32: return PyUnicode_FromString("float");
    case FP64: return PyUnicode_FromString("double");
    case CMPX64: return PyUnicode_FromString("float _Complex");
    case CMPX128: return PyUnicode_FromString("double _Complex");
    case ERROR: {
      PyErr_SetString(PyExc_RuntimeError, "Something unexpected happen, check ./tinytensor/engine/cpu/cpu.c");
      return NULL;
    }
  }
}

static PyObject *getitem(PyObject *self, PyObject *args){
  PyObject *capsule;
  unsigned long long index;
  if(!PyArg_ParseTuple(args, "OK", &capsule, &index)) return NULL;
  if(!PyCapsule_CheckExact(capsule)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid capsule pointer");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(capsule, "tensor_t on CPU");
  if(!tx){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor_t pointer");
    return NULL;
  }
  if(index >= tx->size){
    PyErr_SetString(PyExc_IndexError, "Index out of range");
    return NULL;
  }
  tensor_t *tz = malloc(sizeof(tensor_t));
  if(!tz){
      PyErr_SetString(PyExc_RuntimeError, "tensor allocation failed");
      return NULL;
  }
  tz->dtype = tx->dtype;
  tz->device = tx->device;
  tz->size = 1;
  tz->ndim = 0;
  tz->shape = NULL;
  tz->stride = NULL;
  tz->element_size = tx->element_size;
  tz->storage = malloc(sizeof(storage_t));
  if(!tz->storage){
    free(tz);
    PyErr_SetString(PyExc_RuntimeError, "storage allocation failed");
    return NULL;
  }
  tz->storage->bytes = tz->element_size;
  tz->storage->device = tz->device;
  tz->storage->refcount++;
  tz->storage->ptr = malloc(tz->storage->bytes);
  if(!tz->storage->ptr){
    free(tz->storage);
    free(tz);
    PyErr_SetString(PyExc_RuntimeError, "data allocation failed");
    return NULL;
  }
  tz->buf = tz->storage->ptr;
  char *src = (char*)tx->buf + index * tx->element_size;
  memcpy(tz->buf, src, tz->element_size);
  return PyCapsule_New(tz, "tensor_t on CPU", capsule_destroyer);
}

static PyObject *empty(PyObject *self, PyObject *args){
  PyObject *shape_obj;
  const char *fmt;
  if(!PyArg_ParseTuple(args, "Os", &shape_obj, &fmt)) return NULL;
  if(!PyTuple_Check(shape_obj)){
    PyErr_SetString(PyExc_TypeError, "shape must be a tuple");
    return NULL;
  }
  tensor_t *t = malloc(sizeof(tensor_t));
  if(!t){
    PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed");
    return NULL;
  }
  size_t ndim = PyTuple_Size(shape_obj);
  t->ndim = ndim;
  t->shape  = malloc(ndim * sizeof(size_t));
  t->stride = malloc(ndim * sizeof(size_t));
  if(!t->shape || !t->stride){
    free(t->shape);
    free(t->stride);
    free(t);
    PyErr_SetString(PyExc_RuntimeError, "shape/stride allocation failed");
    return NULL;
  }
  size_t numel = 1;
  for(size_t i = 0; i < ndim; i++){
    size_t dim = PyLong_AsSize_t(PyTuple_GetItem(shape_obj, i));
    if(PyErr_Occurred()){
      free(t->shape);
      free(t->stride);
      free(t);
      return NULL;
    }
    t->shape[i] = dim;
    numel *= dim;
  }
  t->size = numel;
  t->device = (device_t){CPU, 0};
  t->dtype = get_dtype(*fmt);
  if(PyErr_Occurred()){
    free(t->shape);
    free(t->stride);
    free(t);
    return NULL;
  }
  t->element_size = getsize(t->dtype);
  if(ndim > 0){
    t->stride[ndim - 1] = 1;
    for(ssize_t i = ndim - 2; i >= 0; i--){
      t->stride[i] = t->stride[i + 1] * t->shape[i + 1];
    }
  }
  t->storage = malloc(sizeof(storage_t));
  if(!t->storage){
    free(t->shape);
    free(t->stride);
    free(t);
    PyErr_SetString(PyExc_RuntimeError, "storage allocation failed");
    return NULL;
  }
  t->storage->bytes = t->size * t->element_size;
  t->storage->device = t->device;
  t->storage->refcount = 1;
  t->storage->ptr = malloc(t->storage->bytes);
  if(!t->storage->ptr){
    free(t->storage);
    free(t->shape);
    free(t->stride);
    free(t);
    PyErr_SetString(PyExc_RuntimeError, "storage memory allocation failed");
    return NULL;
  }
  t->buf = t->storage->ptr;
  return PyCapsule_New(t, "tensor_t on CPU", capsule_destroyer);
}

static PyMethodDef methods[] = {
  {"tocpu", tocpu, METH_VARARGS, "store tensor in tensor_t"},
  {"topyobj", topyobj, METH_VARARGS, "returns python list"},
  {"ndim", ndim, METH_VARARGS, "returns tensor_t ndim"},
  {"shape", shape, METH_VARARGS, "returns tensor_t shape"},
  {"stride", stride, METH_VARARGS, "returns tensor_t stride"},
  {"device", device, METH_VARARGS, "returns tensor_t device"},
  {"dtype", dtype, METH_VARARGS, "returns tensor_t dtype"},
  {"getitem", getitem, METH_VARARGS, "get item from tensor_t"},
  {"empty", empty, METH_VARARGS, "returns empty tensor_t"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "cpu",
  NULL,
  -1,
  methods
};

PyMODINIT_FUNC PyInit_cpu(void){
  return PyModule_Create(&module);
}
