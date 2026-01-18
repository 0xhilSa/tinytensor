#include "tensor.h"
#include "dtypes.h"
#include <complex.h>
#include <stdint.h>
#include <python3.10/Python.h>

void capsule_destructor(PyObject *capsule){
  tensor_t *t = PyCapsule_GetPointer(capsule, "tensor_t on CPU");
  if(t){
    destroy(t);
    free(t);
  }
}

static PyObject *tocpu(PyObject *self, PyObject *args){
  PyObject *list;
  PyObject *shape_obj;
  const char *fmt;
  if(!PyArg_ParseTuple(args, "OOs", &list, &shape_obj, &fmt)) return NULL;
  if(!PyList_Check(list)){
    PyErr_SetString(PyExc_TypeError, "data must be a list");
    return NULL;
  }
  if(!PyTuple_Check(shape_obj)){
    PyErr_SetString(PyExc_TypeError, "shape must be a tuple");
    return NULL;
  }
  Py_ssize_t ndim = PyTuple_Size(shape_obj);
  if(ndim <= 0){
    PyErr_SetString(PyExc_ValueError, "shape must be non-empty");
    return NULL;
  }
  size_t *shape = malloc(ndim * sizeof(size_t));
  if(!shape){
  PyErr_NoMemory();
  return NULL;
  }
  for(Py_ssize_t i = 0; i < ndim; i++){
    PyObject *item = PyTuple_GetItem(shape_obj, i);
    if(!PyLong_Check(item)){
      free(shape);
      PyErr_SetString(PyExc_TypeError, "shape must contain ints");
      return NULL;
    }
    shape[i] = (size_t)PyLong_AsUnsignedLong(item);
  }
  dtype_t dtype;
  if(strcmp(fmt, "?") == 0) dtype = BOOL;
  else if(strcmp(fmt, "b") == 0) dtype = INT8;
  else if(strcmp(fmt, "B") == 0) dtype = UINT8;
  else if(strcmp(fmt, "h") == 0) dtype = INT16;
  else if(strcmp(fmt, "H") == 0) dtype = UINT16;
  else if(strcmp(fmt, "i") == 0) dtype = INT32;
  else if(strcmp(fmt, "I") == 0) dtype = UINT32;
  else if(strcmp(fmt, "l") == 0) dtype = INT64;
  else if(strcmp(fmt, "L") == 0) dtype = UINT64;
  else if(strcmp(fmt, "f") == 0) dtype = FP32;
  else if(strcmp(fmt, "d") == 0) dtype = FP64;
  else if(strcmp(fmt, "g") == 0) dtype = FP128;
  else if(strcmp(fmt, "F") == 0) dtype = CMPX64;
  else if(strcmp(fmt, "D") == 0) dtype = CMPX128;
  else if(strcmp(fmt, "G") == 0) dtype = CMPX256;
  else { free(shape); PyErr_Format(PyExc_TypeError, "Invalid DType: %s", fmt); return NULL; }

  tensor_t *t = malloc(sizeof(tensor_t));
  if(!t){
    free(t);
    PyErr_NoMemory();
    return NULL;
  }
  device_t device = CPU;
  int device_index = 0;
  *t = create((size_t)ndim, shape, device, device_index, dtype);
  free(shape);
  if(!t->data){
    free(t);
    PyErr_SetString(PyExc_RuntimeError, "tensor allocation failed");
    return NULL;
  }
  if(PyList_Size(list) != (Py_ssize_t)t->length){
    destroy(t);
    free(t);
    PyErr_SetString(PyExc_ValueError, "data size does not match shape");
    return NULL;
  }
  for(Py_ssize_t i = 0; i < (Py_ssize_t)t->length; i++){
    PyObject *item = PyList_GetItem(list, i);
    if(dtype == BOOL){
      i32 v = PyObject_IsTrue(item);
      if(v < 0) goto error;
      ((u8 *)t->data)[i] = (u8)v;
    }else if(dtype == INT8){
      i64 v = PyLong_AsLong(item);
      if(PyErr_Occurred()) goto error;
      ((i8 *)t->data)[i] = (i8)v;
    }else if(dtype == UINT8){
      u64 v = PyLong_AsUnsignedLongMask(item);
      if(PyErr_Occurred()) goto error;
      ((u8 *)t->data)[i] = (u8)v;
    }else if(dtype == INT16){
      i64 v = PyLong_AsLong(item);
      if(PyErr_Occurred()) goto error;
      ((i16 *)t->data)[i] = (i16)v;
    }else if(dtype == UINT16){
      u64 v = PyLong_AsUnsignedLongMask(item);
      if(PyErr_Occurred()) goto error;
      ((u16 *)t->data)[i] = (u16)v;
    }else if(dtype == INT32){
      i64 v = PyLong_AsLong(item);
      if(PyErr_Occurred()) goto error;
      ((i32 *)t->data)[i] = (i32)v;
    }else if(dtype == UINT32){
      u64 v = PyLong_AsUnsignedLongMask(item);
      if(PyErr_Occurred()) goto error;
      ((u32 *)t->data)[i] = (u32)v;
    }else if(dtype == INT64){
      i64 v = PyLong_AsLong(item);
      if(PyErr_Occurred()) goto error;
      ((i64 *)t->data)[i] = (i64)v;
    }else if(dtype == UINT64){
      u64 v = PyLong_AsUnsignedLongMask(item);
      if(PyErr_Occurred()) goto error;
      ((u64 *)t->data)[i] = (u64)v;
    }else if(dtype == FP32){
      f64 v = PyFloat_AsDouble(item);
      if(PyErr_Occurred()) goto error;
      ((f32 *)t->data)[i] = (f32)v;
    }else if(dtype == FP64){
      f64 v = PyFloat_AsDouble(item);
      if(PyErr_Occurred()) goto error;
      ((f64 *)t->data)[i] = (f64)v;
    }else if(dtype == FP128){
      f128 v = PyFloat_AsDouble(item);
      if(PyErr_Occurred()) goto error;
      ((f128 *)t->data)[i] = (f128)v;
    }else if(dtype == CMPX64){
      if(!PyComplex_Check(item)){
        PyErr_SetString(PyExc_TypeError, "expected a complex number");
        goto error;
      }
      f64 real = PyComplex_RealAsDouble(item);
      f64 imag = PyComplex_ImagAsDouble(item);
      if(PyErr_Occurred()) goto error;
      ((c64 *)t->data)[i] = (f32)real + (f32)imag * I;
    }else if(dtype == CMPX128){
      if(!PyComplex_Check(item)){
        PyErr_SetString(PyExc_TypeError, "expected a complex number");
        goto error;
      }
      f64 real = PyComplex_RealAsDouble(item);
      f64 imag = PyComplex_ImagAsDouble(item);
      if(PyErr_Occurred()) goto error;
      ((c128 *)t->data)[i] = real + imag * I;
    }else if(dtype == CMPX256){
      if(!PyComplex_Check(item)){
        PyErr_SetString(PyExc_TypeError, "expected a complex number");
        goto error;
      }
      f128 real = (f128)PyComplex_RealAsDouble(item);
      f128 imag = (f128)PyComplex_ImagAsDouble(item);
      if(PyErr_Occurred()) goto error;
      ((c256 *)t->data)[i] = real + imag * I;
    }else { free(shape); PyErr_Format(PyExc_TypeError, "Invalid DType: %s", fmt); return NULL; }
  }
  return PyCapsule_New(t, "tensor_t on CPU", capsule_destructor);
  error:
    if(shape) free(shape);
    if(t){
      destroy(t);
      free(t);
    }
    return NULL;
}

static PyObject *tolist(PyObject *self, PyObject *args){
  PyObject *capsule;
  if(!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
  tensor_t *t = PyCapsule_GetPointer(capsule, "tensor_t on CPU");
  if(!t){
    PyErr_SetString(PyExc_ValueError, "Invalid CPU tensor capsule");
    return NULL;
  }
  if(!t->data){
    PyErr_SetString(PyExc_RuntimeError, "Tensor has no data");
    return NULL;
  }
  PyObject *list = PyList_New(t->length);
  if(!list) return NULL;
  for(size_t i = 0; i < t->length; i++){
    PyObject *item = NULL;
    dtype_t dtype= t->dtype;
    if(dtype == BOOL) item = PyBool_FromLong(((u8 *)t->data)[i]);
    else if(dtype == INT8) item = PyLong_FromLong(((i8 *)t->data)[i]);
    else if(dtype == UINT8) item = PyLong_FromUnsignedLong(((u8 *)t->data)[i]);
    else if(dtype == INT16) item = PyLong_FromLong(((i16 *)t->data)[i]);
    else if(dtype == UINT16) item = PyLong_FromUnsignedLong(((u16 *)t->data)[i]);
    else if(dtype == INT32) item = PyLong_FromLong(((i32 *)t->data)[i]);
    else if(dtype == UINT32) item = PyLong_FromUnsignedLong(((u32 *)t->data)[i]);
    else if(dtype == INT64) item = PyLong_FromLong(((i64 *)t->data)[i]);
    else if(dtype == UINT64) item = PyLong_FromUnsignedLong(((u64 *)t->data)[i]);
    else if(dtype == FP32) item = PyFloat_FromDouble((f64)((f32 *)t->data)[i]);
    else if(dtype == FP64) item = PyFloat_FromDouble(((f64 *)t->data)[i]);
    else if(dtype == FP128) item = PyFloat_FromDouble((f64)((f128 *)t->data)[i]);
    else if(dtype == CMPX64){
      c64 v = ((c64 *)t->data)[i];
      item = PyComplex_FromDoubles(crealf(v), cimagf(v));
    }else if(dtype == CMPX128){
      c128 v = ((c128 *)t->data)[i];
      item = PyComplex_FromDoubles(creal(v), cimag(v));
    }else if(dtype == CMPX256){
      c256 v = ((c256 *)t->data)[i];
      item = PyComplex_FromDoubles((double)creall(v), (double)cimagl(v));
    }else{
      Py_DECREF(list);
      PyErr_SetString(PyExc_RuntimeError, "Unknown dtype");
      return NULL;
    }
    if(!item){
      Py_DECREF(list);
      return NULL;
    }
    PyList_SET_ITEM(list, i, item);
  }
  return list;
}

static PyMethodDef methods[] = {
  {"tocpu", tocpu, METH_VARARGS, "store tensor"},
  {"tolist", tolist, METH_VARARGS, "return tensor into the python list"},
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
