#include <cuda_runtime.h>
#include <cuComplex.h>
#include <python3.10/Python.h>
#include "../tensor.h"

static void capsule_destructor(PyObject *capsule){
  tensor_t *t = (tensor_t *)PyCapsule_GetPointer(capsule, "tensor_t on CUDA");
  if(!t) return;
  destroy(t);
  free(t);
}

static void cpu_capsule_destructor(PyObject *capsule){
  tensor_t *t = (tensor_t *)PyCapsule_GetPointer(capsule, "tensor_t on CPU");
  if(!t) return;
  destroy(t);
  free(t);
}

static PyObject *tocuda(PyObject *self, PyObject *args){
  PyObject *list;
  PyObject *shape_obj;
  const char *fmt;
  int device_index;
  if(!PyArg_ParseTuple(args, "OOsi", &list, &shape_obj, &fmt, &device_index)) return NULL;
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
  size_t *shape = (size_t *)malloc(ndim * sizeof(size_t));
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

  tensor_t *t = (tensor_t *)malloc(sizeof(tensor_t));
  if(!t){
    free(shape);
    PyErr_NoMemory();
    return NULL;
  }
  device_t device = CUDA;
  *t = create((size_t)ndim, shape, device, device_index, dtype);
  free(shape);
  if(PyList_Size(list) != (Py_ssize_t)t->length){
    destroy(t);
    free(t);
    PyErr_SetString(PyExc_ValueError, "data size does not match shape");
    return NULL;
  }
  size_t bytes = t->length * t->elem_size;
  void *dptr = NULL;
  int cu_device_index = t->device_index;
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if(err != cudaSuccess){
    destroy(t);
    free(t);
    PyErr_SetString(PyExc_RuntimeError, cudaGetErrorString(err));
    return NULL;
  }
  if(cu_device_index >= 0 && cu_device_index < count) cudaSetDevice(cu_device_index);
  else{
    destroy(t);
    free(t);
    PyErr_Format(PyExc_RuntimeError, "Invalid device index: %i", cu_device_index);
    return NULL;
  }
  err = cudaMalloc(&dptr, bytes);
  if(err != cudaSuccess){
    destroy(t);
    free(t);
    PyErr_SetString(PyExc_RuntimeError, cudaGetErrorString(err));
    return NULL;
  }
  void *hbuf = malloc(bytes);
  if(!hbuf){
    cudaFree(dptr);
    destroy(t);
    free(t);
    PyErr_NoMemory();
    return NULL;
  }
  for(Py_ssize_t i = 0; i < (Py_ssize_t)t->length; i++){
    PyObject *item = PyList_GetItem(list, i);
    if(dtype == BOOL) ((u8 *)hbuf)[i] = (u8)PyObject_IsTrue(item);
    else if(dtype == INT8) ((i8 *)hbuf)[i] = (i8)PyLong_AsLong(item);
    else if(dtype == UINT8) ((u8 *)hbuf)[i] = (u8)PyLong_AsUnsignedLongMask(item);
    else if(dtype == INT16) ((i16 *)hbuf)[i] = (i16)PyLong_AsLong(item);
    else if(dtype == UINT16) ((u16 *)hbuf)[i] = (u16)PyLong_AsUnsignedLongMask(item);
    else if(dtype == INT32) ((i32 *)hbuf)[i] = (i32)PyLong_AsLong(item);
    else if(dtype == UINT32) ((u32 *)hbuf)[i] = (u32)PyLong_AsUnsignedLongMask(item);
    else if(dtype == INT64) ((i64 *)hbuf)[i] = (i64)PyLong_AsLong(item);
    else if(dtype == UINT64) ((u64 *)hbuf)[i] = (u64)PyLong_AsUnsignedLongMask(item);
    else if(dtype == FP32) ((f32 *)hbuf)[i] = (f32)PyFloat_AsDouble(item);
    else if(dtype == FP64) ((f64 *)hbuf)[i] = (f64)PyFloat_AsDouble(item);
    else if(dtype == FP128) ((f128 *)hbuf)[i] = (f128)PyFloat_AsDouble(item);
    else if(dtype == CMPX64){
      if(!PyComplex_Check(item)){
        PyErr_SetString(PyExc_TypeError, "expected a complex number");
        goto error;
      }
      float real = (float)PyComplex_RealAsDouble(item);
      float imag = (float)PyComplex_ImagAsDouble(item);
      ((cuFloatComplex *)hbuf)[i] = make_cuFloatComplex(real, imag);
    }else if(dtype == CMPX128){
      if(!PyComplex_Check(item)){
        PyErr_SetString(PyExc_TypeError, "expected a complex number");
        goto error;
      }
      double real = (double)PyComplex_RealAsDouble(item);
      double imag = (double)PyComplex_ImagAsDouble(item);
      ((cuDoubleComplex *)hbuf)[i] = make_cuDoubleComplex(real, imag);
    }else if(dtype == CMPX256){
      PyErr_SetString(PyExc_NotImplementedError, "complex256 is not supported by cuComplex");
      goto error;
    }
  }
  if(PyErr_Occurred()) goto error;
  err = cudaMemcpy(dptr, hbuf, bytes, cudaMemcpyHostToDevice);
  free(hbuf);

  if(err != cudaSuccess){
    cudaFree(dptr);
    destroy(t);
    free(t);
    PyErr_SetString(PyExc_RuntimeError, cudaGetErrorString(err));
    return NULL;
  }
  t->data = dptr;
  return PyCapsule_New(t, "tensor_t on CUDA", capsule_destructor);

  error:
    free(hbuf);
    cudaFree(dptr);
    if(shape) free(shape);
      if(t){
        destroy(t);
        free(t);
      }
    return NULL;
}

static PyObject *tocpu(PyObject *self, PyObject *args){
  PyObject *capsule;
  if(!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
  tensor_t *src = (tensor_t *)PyCapsule_GetPointer(capsule, "tensor_t on CUDA");
  if(!src){
    PyErr_SetString(PyExc_ValueError, "Invalid CUDA tensor capsule");
    return NULL;
  }
  tensor_t *dst = (tensor_t *)malloc(sizeof(tensor_t));
  if(!dst){
    PyErr_NoMemory();
    return NULL;
  }
  device_t device = CPU;
  int device_index = 0; // switching to CPU
  *dst = create(src->ndim, src->shape, device, device_index, src->dtype);
  size_t bytes = src->length * src->elem_size;
  dst->data = malloc(bytes);
  if(!dst->data){
    destroy(dst);
    free(dst);
    PyErr_NoMemory();
    return NULL;
  }
  int cu_device_index = src->device_index;
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
      destroy(dst);
      free(dst);
      PyErr_SetString(PyExc_RuntimeError, cudaGetErrorString(err));
      return NULL;
  }
  if(cu_device_index >= 0 && cu_device_index < count) cudaSetDevice(cu_device_index);
  else{
    destroy(dst);
    free(dst);
    PyErr_Format(PyExc_RuntimeError, "Invalid device index: %i", cu_device_index);
    return NULL;
  }
  err = cudaMemcpy(
    dst->data,
    src->data,
    bytes,
    cudaMemcpyDeviceToHost
  );
  if(err != cudaSuccess){
    destroy(dst);
    free(dst);
    PyErr_SetString(PyExc_RuntimeError, cudaGetErrorString(err));
    return NULL;
  }
  return PyCapsule_New(dst, "tensor_t on CPU", cpu_capsule_destructor);
}

static PyObject *device_count(PyObject *self, PyObject *args){
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if(err != cudaSuccess){
    PyErr_SetString(PyExc_RuntimeError, cudaGetErrorString(err) );
    return NULL;
  }
  return PyLong_FromLong((long)count);
}

static PyObject *device_name(PyObject *self, PyObject *args){
  int device;
  if(!PyArg_ParseTuple(args, "i", &device)) return NULL;
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if(device < 0 || device >= device_count){
    PyErr_SetString(PyExc_ValueError, "Invalid CUDA device index");
    return NULL;
  }
  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, device);
  if(err != cudaSuccess){
    PyErr_SetString(PyExc_RuntimeError, cudaGetErrorString(err));
    return NULL;
  }
  return PyUnicode_FromString(prop.name);
}

static PyObject *is_available(PyObject *self, PyObject *args){
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if(err == cudaErrorNoDevice) Py_RETURN_FALSE;
  if(err != cudaSuccess) Py_RETURN_FALSE;
  if(count > 0) Py_RETURN_TRUE;
  else Py_RETURN_FALSE;
}

static PyObject *get_device(PyObject *self, PyObject *args){
  int device;
  cudaError_t err = cudaGetDevice(&device);
  if(err != cudaSuccess){
    PyErr_SetString(PyExc_RuntimeError, cudaGetErrorString(err));
    return NULL;
  }
  return PyLong_FromLong(device);
}

static PyObject *device_prop(PyObject *self, PyObject *args){
  int device;
  if(!PyArg_ParseTuple(args, "i", &device)) return NULL;
  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, device);
  if(err != cudaSuccess){
    PyErr_SetString(PyExc_RuntimeError, cudaGetErrorString(err));
    return NULL;
  }
  PyObject *dict = PyDict_New();
  if(!dict) return NULL;
  PyDict_SetItemString(dict, "name", PyUnicode_FromString(prop.name));
  PyDict_SetItemString(dict, "totalGlobalMem", PyLong_FromUnsignedLongLong(prop.totalGlobalMem));
  PyDict_SetItemString(dict, "sharedMemPerBlock", PyLong_FromUnsignedLongLong(prop.sharedMemPerBlock));
  PyDict_SetItemString(dict, "regsPerBlock", PyLong_FromLong(prop.regsPerBlock));
  PyDict_SetItemString(dict, "warpSize", PyLong_FromLong(prop.warpSize));
  PyDict_SetItemString(dict, "memPitch", PyLong_FromUnsignedLongLong(prop.memPitch));
  PyDict_SetItemString(dict, "maxThreadsPerBlock", PyLong_FromLong(prop.maxThreadsPerBlock));
  PyDict_SetItemString(dict, "multiProcessorCount", PyLong_FromLong(prop.multiProcessorCount));
  PyDict_SetItemString(dict, "clockRate", PyLong_FromLong(prop.clockRate));
  PyDict_SetItemString(dict, "major", PyLong_FromLong(prop.major));
  PyDict_SetItemString(dict, "minor", PyLong_FromLong(prop.minor));
  PyDict_SetItemString(dict, "integrated", PyBool_FromLong(prop.integrated));
  PyDict_SetItemString(dict, "canMapHostMemory", PyBool_FromLong(prop.canMapHostMemory));
  PyDict_SetItemString(dict, "concurrentKernels", PyBool_FromLong(prop.concurrentKernels));
  PyDict_SetItemString(dict, "ECCEnabled", PyBool_FromLong(prop.ECCEnabled));
  return dict;
}

static PyMethodDef methods[] = {
  {"tocuda", tocuda, METH_VARARGS, "store tensor on CUDA"},
  {"tocpu", tocpu, METH_VARARGS, "copy tensor from CUDA to CPU"},
  {"device_count", device_count, METH_NOARGS, "return the number of CUDA device"},
  {"device_name", device_name, METH_VARARGS, "return the name of the CUDA device"},
  {"is_available", is_available, METH_NOARGS, "return if the CUDA device is present or not?"},
  {"get_device", get_device, METH_NOARGS, "get CUDA device"},
  {"device_prop", device_prop, METH_VARARGS, "return device properties"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "cuda",
  NULL,
  -1,
  methods
};

PyMODINIT_FUNC PyInit_cuda(void){
  return PyModule_Create(&module);
}
