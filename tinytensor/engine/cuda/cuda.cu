#include <cuda_runtime.h>
#include <python3.10/Python.h>
#include <nvml.h>
#include <python3.10/pyerrors.h>
#include "../tensor.h"

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
      PyErr_Format(PyExc_RuntimeError, "CUDA error: %s", cudaGetErrorString(err)); \
      return NULL; \
    } \
  } while(0)

void capsule_destroyer(PyObject *capsule){
  tensor_t *t = (tensor_t *)PyCapsule_GetPointer(capsule, "tensor_t on CPU");
  if(t) destroy(t);
}

void capsule_destroyer_cuda(PyObject *capsule){
  tensor_t *t = (tensor_t *)PyCapsule_GetPointer(capsule, "tensor_t on CUDA");
  if(t) destroy(t);
}

static inline long long py_to_int(PyObject *o){ return PyLong_AsLongLong(o); }
static inline unsigned long long py_to_uint(PyObject *o){ return PyLong_AsUnsignedLongLongMask(o); }
static inline double py_to_float(PyObject *o){ return PyFloat_AsDouble(o); }

dtype_t get_dtype(const char *fmt){
  switch(*fmt){
    case '?': return BOOL;
    case 'b': return INT8;
    case 'B': return UINT8;
    case 'h': return INT16;
    case 'H': return UINT16;
    case 'i': return INT32;
    case 'I': return UINT32;
    case 'l': return INT64;
    case 'L': return UINT64;
    case 'f': return FP32;
    case 'd': return FP64;
    case 'F': return CMPX64;
    case 'D': return CMPX128;
    default: return ERROR;
  }
}

static PyObject *__list__(PyObject *list, PyObject *shape, const char *fmt, int device_idx){
  size_t length = (size_t)PyList_Size(list);
  dtype_t dtype = get_dtype(fmt);
  if(dtype == ERROR){
    PyErr_Format(PyExc_TypeError, "Invalid dtype fmt: '%c'", fmt);
    return NULL;
  }
  int device_count;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA error: %s", cudaGetErrorString(err));
    return NULL;
  }
  if(device_idx < 0 || device_idx >= device_count){
    PyErr_Format(PyExc_ValueError, "Invalid CUDA device index: %d (available: 0-%d)", device_idx, device_count - 1);
    return NULL;
  }
  CUDA_CHECK(cudaSetDevice(device_idx));
  tensor_t *t = (tensor_t *)malloc(sizeof(tensor_t));
  if(!t){
    PyErr_SetString(PyExc_RuntimeError, "tensor_t allocation failed!");
    return NULL;
  }
  t->dtype = dtype;
  t->size = length;
  t->ndim = PyTuple_Size(shape);
  t->shape = (size_t *)malloc(sizeof(size_t) * t->ndim);
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
  t->stride = (size_t *)malloc(sizeof(size_t) * t->ndim);
  if(!t->stride){
    destroy(t);
    PyErr_SetString(PyExc_RuntimeError, "tensor_t.stride allocation failed!");
    return NULL;
  }
  t->stride[t->ndim-1] = 1;
  for(int i = (int)t->ndim - 2; i >= 0; i--){ t->stride[i] = t->shape[i + 1] * t->stride[i+1]; }
  t->element_size = getsize(dtype);
  t->device = (device_t){CUDA, (unsigned short)device_idx};
  t->storage = (storage_t *)malloc(sizeof(storage_t));
  if(!t->storage){
    free(t->shape);
    free(t);
    PyErr_SetString(PyExc_RuntimeError, "tensor_t.storage allocation failed!");
    return NULL;
  }
  t->storage->bytes = t->size * t->element_size;
  t->storage->device = t->device;
  t->storage->refcount = 1;
  err = cudaMalloc(&t->storage->ptr, t->storage->bytes);
  if(err != cudaSuccess){
    free(t->storage);
    free(t->shape);
    free(t);
    PyErr_Format(PyExc_RuntimeError, "CUDA memory allocation failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  t->buf = t->storage->ptr;
  void *host_buffer = malloc(t->storage->bytes);
  if(!host_buffer){
    cudaFree(t->storage->ptr);
    free(t->storage);
    free(t->shape);
    free(t);
    PyErr_SetString(PyExc_RuntimeError, "Host staging buffer allocation failed!");
    return NULL;
  }
  for(Py_ssize_t i = 0; i < length; i++){
    PyObject *item = PyList_GetItem(list, i);
    char *p = (char *)host_buffer + i * t->element_size;
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
      case FP32: *(float32 *)p = (float32)py_to_float(item); break;
      case FP64: *(float64 *)p = (float64)py_to_float(item); break;
      case CMPX64: {
        Py_complex c = PyComplex_AsCComplex(item);
        if(PyErr_Occurred()){
          free(host_buffer);
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
          free(host_buffer);
          destroy(t);
          return NULL;
        }
        ((complex128 *)p)->real = (float64)c.real;
        ((complex128 *)p)->imag = (float64)c.imag;
        break;
      }
      case ERROR: {
        free(host_buffer);
        destroy(t);
        PyErr_Format(PyExc_TypeError, "Invalid dtype fmt: '%c'", fmt);
        return NULL;
      }
    }
  }
  err = cudaMemcpy(t->storage->ptr, host_buffer, t->storage->bytes, cudaMemcpyHostToDevice);
  free(host_buffer);
  if(err != cudaSuccess){
    destroy(t);
    PyErr_Format(PyExc_RuntimeError, "CUDA memcpy failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  return PyCapsule_New(t, "tensor_t on CUDA", capsule_destroyer_cuda);
}

static PyObject *__scalar__(PyObject *scalar, const char *fmt, int device_idx){
  if(PyList_Check(scalar)){
    if(PyList_Size(scalar) != 1){
      PyErr_SetString(PyExc_TypeError, "Scalar tensor expects exactly one element");
      return NULL;
    }
    scalar = PyList_GetItem(scalar, 0);  /* borrowed ref */
  }
  dtype_t dtype = get_dtype(fmt);
  if(dtype == ERROR){
    PyErr_Format(PyExc_TypeError, "Invalid dtype fmt: '%c'", fmt);
    return NULL;
  }
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA error: %s", cudaGetErrorString(err));
    return NULL;
  }
  if(device_idx < 0 || device_idx >= device_count) {
    PyErr_Format(PyExc_ValueError, "Invalid CUDA device index: %d (available: 0-%d)", device_idx, device_count - 1);
    return NULL;
  }
  CUDA_CHECK(cudaSetDevice(device_idx));
  tensor_t *t = (tensor_t *)malloc(sizeof(tensor_t));
  if(!t){
    PyErr_NoMemory();
    return NULL;
  }
  t->dtype = dtype;
  t->size = 1;
  t->ndim = 0;
  t->shape = NULL;
  t->stride = NULL;
  t->element_size = getsize(dtype);
  t->device = (device_t){ CUDA, (unsigned short)device_idx };
  t->storage = (storage_t *)malloc(sizeof(storage_t));
  if(!t->storage){
    free(t);
    PyErr_NoMemory();
    return NULL;
  }
  t->storage->bytes = t->element_size;
  t->storage->device = t->device;
  t->storage->refcount = 1;
  err = cudaMalloc(&t->storage->ptr, t->storage->bytes);
  if(err != cudaSuccess){
    free(t->storage);
    free(t);
    PyErr_Format(PyExc_RuntimeError, "CUDA memory allocation failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  t->buf = t->storage->ptr;
  void *host_buffer = malloc(t->element_size);
  if(!host_buffer){
    cudaFree(t->storage->ptr);
    free(t->storage);
    free(t);
    PyErr_NoMemory();
    return NULL;
  }
  void *p = host_buffer;
  switch(dtype){
    case BOOL: {
      int v = PyObject_IsTrue(scalar);
      if(v < 0) goto fail;
      *(bool *)p = (bool)v;
      break;
    }
    case INT8:
    case INT16:
    case INT32:
    case INT64: {
      long long v = PyLong_AsLongLong(scalar);
      if(PyErr_Occurred()) goto fail;
      memcpy(p, &v, t->element_size);
      break;
    }
    case UINT8:
    case UINT16:
    case UINT32:
    case UINT64: {
      unsigned long long v = PyLong_AsUnsignedLongLongMask(scalar);
      if(PyErr_Occurred()) goto fail;
      memcpy(p, &v, t->element_size);
      break;
    }
    case FP32:
    case FP64: {
      double v = PyFloat_AsDouble(scalar);
      if(PyErr_Occurred()) goto fail;
      if(dtype == FP32) *(float *)p = (float)v;
      else *(double *)p = v;
      break;
    }
    case CMPX64: {
      Py_complex c = PyComplex_AsCComplex(scalar);
      if(PyErr_Occurred()) goto fail;
      ((complex64 *)p)->real = (float)c.real;
      ((complex64 *)p)->imag = (float)c.imag;
      break;
    }
    case CMPX128: {
      Py_complex c = PyComplex_AsCComplex(scalar);
      if(PyErr_Occurred()) goto fail;
      ((complex128 *)p)->real = c.real;
      ((complex128 *)p)->imag = c.imag;
      break;
    }
    default:
      PyErr_SetString(PyExc_TypeError, "Unsupported dtype");
      goto fail;
  }
  err = cudaMemcpy(t->storage->ptr, host_buffer, t->element_size, cudaMemcpyHostToDevice);
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "CUDA memcpy failed: %s", cudaGetErrorString(err));
    goto fail;
  }
  free(host_buffer);
  return PyCapsule_New(t, "tensor_t on CUDA", capsule_destroyer_cuda);
fail:
  free(host_buffer);
  destroy(t);
  return NULL;
}

static PyObject *tocuda(PyObject *self, PyObject *args){
  PyObject *pyobj; // array or scalar
  PyObject *shape; // empty for scalar
  const char *fmt;
  int device_idx;
  if(!PyArg_ParseTuple(args, "OOsi", &pyobj, &shape, &fmt, &device_idx)) return NULL;
  Py_ssize_t ndim = PyTuple_Size(shape);
  if(ndim != 0) return __list__(pyobj, shape, fmt, device_idx);
  else if(ndim == 0) return __scalar__(pyobj, fmt, device_idx);
  else{
    PyErr_SetString(PyExc_RuntimeError, "Something is wrong i can feel it!");
    return NULL;
  }
}

static PyObject *topyobj(PyObject *self, PyObject *args){
  PyObject *capsule;
  if(!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
  tensor_t *t = (tensor_t *)PyCapsule_GetPointer(capsule, "tensor_t on CUDA");
  if(!t){
    PyErr_SetString(PyExc_TypeError, "Invalid tensor capsule");
    return NULL;
  }
  if(t->device.type != CUDA){
    PyErr_SetString(PyExc_TypeError, "Tensor is not on CUDA device");
    return NULL;
  }
  cudaError_t err = cudaSetDevice(t->device.index);
  if(err != cudaSuccess){
    PyErr_Format(PyExc_RuntimeError, "Failed to set CUDA device: %s", cudaGetErrorString(err));
    return NULL;
  }
  void *host_buffer = malloc(t->storage->bytes);
  if(!host_buffer){
    PyErr_NoMemory();
    return NULL;
  }
  err = cudaMemcpy(host_buffer, t->storage->ptr, t->storage->bytes, cudaMemcpyDeviceToHost);
  if(err != cudaSuccess){
    free(host_buffer);
    PyErr_Format(PyExc_RuntimeError, "CUDA memcpy failed: %s", cudaGetErrorString(err));
    return NULL;
  }
  // if its a scalar
  if(t->size == 1 && !t->shape){
    switch(t->dtype){
      case BOOL: return PyBool_FromLong(*(bool *)host_buffer);
      case INT8: return PyLong_FromLongLong(*(int8 *)host_buffer);
      case UINT8: return PyLong_FromUnsignedLongLong(*(uint8 *)host_buffer);
      case INT16: return PyLong_FromLongLong(*(int16 *)host_buffer);
      case UINT16: return PyLong_FromUnsignedLongLong(*(uint16 *)host_buffer);
      case INT32: return PyLong_FromLongLong(*(int32 *)host_buffer);
      case UINT32: return PyLong_FromUnsignedLongLong(*(uint32 *)host_buffer);
      case INT64: return PyLong_FromLongLong(*(int64 *)host_buffer);
      case UINT64: return PyLong_FromUnsignedLongLong(*(uint64 *)host_buffer);
      case FP32: return PyFloat_FromDouble(*(float32 *)host_buffer);
      case FP64: return PyFloat_FromDouble(*(float64 *)host_buffer);
      case CMPX64: {
        complex64 *c = (complex64 *)host_buffer;
        return PyComplex_FromDoubles((double)c->real, (double)c->imag);
      }
      case CMPX128: {
        complex128 *c = (complex128 *)host_buffer;
        return PyComplex_FromDoubles(c->real, c->imag);
      }
      default:
        free(host_buffer);
        PyErr_Format(PyExc_TypeError, "Unsupported dtype: %d", t->dtype);
        return NULL;
      free(host_buffer);
    }
  }else if(t->shape){
    PyObject *list = PyList_New(t->size);
    if(!list){
      free(host_buffer);
      return NULL;
    }
    for(size_t i = 0; i < t->size; i++){
      char *p = (char *)host_buffer + i * t->element_size;
      PyObject *item = NULL;
      switch(t->dtype){
        case BOOL:
          item = PyBool_FromLong(*(bool *)p);
          break;
        case INT8:
          item = PyLong_FromLongLong(*(int8 *)p);
          break;
        case UINT8:
          item = PyLong_FromUnsignedLongLong(*(uint8 *)p);
          break;
        case INT16:
          item = PyLong_FromLongLong(*(int16 *)p);
          break;
        case UINT16:
          item = PyLong_FromUnsignedLongLong(*(uint16 *)p);
          break;
        case INT32:
          item = PyLong_FromLongLong(*(int32 *)p);
          break;
        case UINT32:
          item = PyLong_FromUnsignedLongLong(*(uint32 *)p);
          break;
        case INT64:
          item = PyLong_FromLongLong(*(int64 *)p);
          break;
        case UINT64:
          item = PyLong_FromUnsignedLongLong(*(uint64 *)p);
          break;
        case FP32:
          item = PyFloat_FromDouble(*(float32 *)p);
          break;
        case FP64:
          item = PyFloat_FromDouble(*(float64 *)p);
          break;
        case CMPX64: {
          complex64 *c = (complex64 *)p;
          item = PyComplex_FromDoubles((double)c->real, (double)c->imag);
          break;
        }
        case CMPX128: {
          complex128 *c = (complex128 *)p;
          item = PyComplex_FromDoubles(c->real, c->imag);
          break;
        }
        default:
          free(host_buffer);
          Py_DECREF(list);
          PyErr_Format(PyExc_TypeError, "Unsupported dtype: %d", t->dtype);
          return NULL;
      }
      if(!item){
        free(host_buffer);
        Py_DECREF(list);
        return NULL;
      }
      PyList_SET_ITEM(list, i, item);  // Steals reference to item
    }
    free(host_buffer);
    return list;
  }else{
    PyErr_SetString(PyExc_RuntimeError, "Condition didn't satisfied!");
    return NULL;
  }
}

static PyObject *device_name(PyObject *self, PyObject *args){
  int device;
  if(!PyArg_ParseTuple(args, "i", &device)) return NULL;
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if(err != cudaSuccess){
    PyErr_SetString(PyExc_RuntimeError, cudaGetErrorString(err));
    return NULL;
  }
  if(device < 0 || device >= device_count){
    PyErr_SetString(PyExc_ValueError, "Invalid CUDA device index");
    return NULL;
  }
  cudaDeviceProp prop;
  err = cudaGetDeviceProperties(&prop, device);
  if(err != cudaSuccess){
    PyErr_SetString(PyExc_RuntimeError, cudaGetErrorString(err));
    return NULL;
  }
  return PyUnicode_FromString(prop.name);
}

static PyObject *device_count(PyObject *self, PyObject *args){
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if(err){
    PyErr_SetString(PyExc_RuntimeError, cudaGetErrorString(err));
    return NULL;
  }
  return PyLong_FromLong(count);
}

static PyObject *is_available(PyObject *self, PyObject *args){
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if(err == cudaErrorNoDevice) Py_RETURN_FALSE;
  if(err != cudaSuccess) Py_RETURN_FALSE;
  if(count > 0) Py_RETURN_TRUE;
  else Py_RETURN_FALSE;
}

static PyObject *get_device_prop(PyObject *self, PyObject *args){
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
#if CUDART_VERSION < 13000
  PyDict_SetItemString(dict, "clockRate", PyLong_FromLong(prop.clockRate));
#else
    PyDict_SetItemString(dict, "clockRate", PyLong_FromLong(-1));  // Not supported in CUDA 13+
#endif
  PyDict_SetItemString(dict, "major", PyLong_FromLong(prop.major));
  PyDict_SetItemString(dict, "minor", PyLong_FromLong(prop.minor));
  PyDict_SetItemString(dict, "integrated", PyBool_FromLong(prop.integrated));
  PyDict_SetItemString(dict, "canMapHostMemory", PyBool_FromLong(prop.canMapHostMemory));
  PyDict_SetItemString(dict, "concurrentKernels", PyBool_FromLong(prop.concurrentKernels));
  PyDict_SetItemString(dict, "ECCEnabled", PyBool_FromLong(prop.ECCEnabled));
  return dict;
}

static PyObject *runtime_version(PyObject *self, PyObject *args){
  int v = 0;
  cudaError_t err = cudaRuntimeGetVersion(&v);
  if(err != cudaSuccess){
    PyErr_SetString(PyExc_RuntimeError, cudaGetErrorString(err));
    return NULL;
  }
  int major = v / 1000;
  int minor = (v % 1000) / 10;
  return Py_BuildValue("(ii)", major, minor);
}

static PyObject *driver_version(PyObject *self, PyObject *args){
  int v = 0;
  cudaError_t err = cudaDriverGetVersion(&v);
  if(err != cudaSuccess){
    PyErr_SetString(PyExc_RuntimeError, cudaGetErrorString(err));
    return NULL;
  }
  int major = v / 1000;
  int minor = (v % 1000) / 10;
  return Py_BuildValue("(ii)", major, minor);
}

static PyObject *driver_package(PyObject *self, PyObject *args){
  nvmlReturn_t res = nvmlInit();
  if(res != NVML_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, nvmlErrorString(res));
    return NULL;
  }
  char version[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
  res = nvmlSystemGetDriverVersion(version, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE);
  nvmlShutdown();
  if(res != NVML_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, nvmlErrorString(res));
    return NULL;
  }
  return PyUnicode_FromString(version);
}

static PyObject *get_device(PyObject *self, PyObject *args){
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if(err != cudaSuccess){
    PyErr_SetString(PyExc_RuntimeError, cudaGetErrorString(err));
    return NULL;
  }
  return PyLong_FromLong(device);
}

static PyObject *ndim(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  tensor_t *t = (tensor_t *)PyCapsule_GetPointer(x, "tensor_t on CUDA");
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
  tensor_t *t = (tensor_t *)PyCapsule_GetPointer(x, "tensor_t on CUDA");
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
  tensor_t *t = (tensor_t *)PyCapsule_GetPointer(x, "tensor_t on CUDA");
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
  tensor_t *t = (tensor_t *)PyCapsule_GetPointer(x, "tensor_t on CUDA");
  if(!t){
    PyErr_SetString(PyExc_RuntimeError, "invalid tensor capsule");
    return NULL;
  }
  const char *type = (t->device.type == CUDA) ? "CUDA" : "CPU";
  printf("device: %s\n", type);
  return Py_BuildValue("(si)", type, t->device.index);
}

static PyObject *dtype(PyObject *self, PyObject *args){
  PyObject *x;
  if(!PyArg_ParseTuple(args, "O", &x)) return NULL;
  if(!PyCapsule_CheckExact(x)){
    PyErr_SetString(PyExc_TypeError, "expected tensor capsule");
    return NULL;
  }
  tensor_t *t = (tensor_t *)PyCapsule_GetPointer(x, "tensor_t on CUDA");
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
    case FP32: return PyUnicode_FromString("float");
    case FP64: return PyUnicode_FromString("double");
    case CMPX64: return PyUnicode_FromString("float _Complex");
    case CMPX128: return PyUnicode_FromString("double _Complex");
    default: {
      PyErr_SetString(PyExc_RuntimeError, "Something unexpected happen, check ./tinytensor/engine/cpu/cpu.c");
      return NULL;
    }
  }
}

static PyMethodDef methods[] = {
  {"tocuda", tocuda, METH_VARARGS, "returns CUDA tensor_t capsule"},
  {"topyobj", topyobj, METH_VARARGS, "returns CPU tensor_t capsule from CUDA tensor_t"},
  {"device_name", device_name, METH_VARARGS, "returns the device name"},
  {"device_count", device_count, METH_NOARGS, "returns the device count"},
  {"is_available", is_available, METH_NOARGS, "returns if cuda device available"},
  {"get_device_prop", get_device_prop, METH_VARARGS, "returns the device property"},
  {"runtime_version", runtime_version, METH_NOARGS, "returns runtime version"},
  {"driver_version", driver_version, METH_NOARGS, "returns driver version"},
  {"driver_package", driver_package, METH_NOARGS, "returns driver package version"},
  {"get_device", get_device, METH_VARARGS, "returns which cuda device is being use"},
  {"ndim", ndim, METH_VARARGS, "returns tensor_t ndim"},
  {"shape", shape, METH_VARARGS, "returns tensor_t shape"},
  {"stride", stride, METH_VARARGS, "returns tensor_t stride"},
  {"device", device, METH_VARARGS, "returns tensor_t device"},
  {"dtype", dtype, METH_VARARGS, "returns tensor_t dtype"},
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
