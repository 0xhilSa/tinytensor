#include <python3.10/Python.h>
#include <time.h>
#include <stdint.h>
#include <stdlib.h>
#include "../tensor.h"

#define PHILOX_M0 0xd2511f53
#define PHILOX_M1 0xcd9e8d57
#define PHILOX_W0 0x9e3779b9
#define PHILOX_W1 0xbb67ae85

static uint64_t global_seed = 0;

void capsule_destroyer(PyObject *capsule){
  tensor_t *t = (tensor_t *)PyCapsule_GetPointer(capsule, "tensor_t on CPU");
  if(t) destroy(t);
}

static inline void mulhilo(uint32_t a, uint32_t b, uint32_t *hi, uint32_t *lo){
  uint64_t p = (uint64_t)a * b;
  *lo = (uint32_t)p;
  *hi = (uint32_t)(p >> 32);
}

static inline void philox_round(uint32_t ctr[4], uint32_t key[2]){
  uint32_t hi0, lo0;
  uint32_t hi1, lo1;
  mulhilo(PHILOX_M0, ctr[0], &hi0, &lo0);
  mulhilo(PHILOX_M1, ctr[2], &hi1, &lo1);
  uint32_t c0 = hi1 ^ ctr[1] ^ key[0];
  uint32_t c1 = lo1;
  uint32_t c2 = hi0 ^ ctr[3] ^ key[1];
  uint32_t c3 = lo0;
  ctr[0] = c0;
  ctr[1] = c1;
  ctr[2] = c2;
  ctr[3] = c3;
}

static inline void philox4x32(uint32_t ctr[4], uint32_t key[2]){
  for(int i=0;i<10;i++){
    philox_round(ctr,key);
    key[0] += PHILOX_W0;
    key[1] += PHILOX_W1;
  }
}

static inline float uint32_to_uniform(uint32_t x){
  return x * (1.0f / 4294967296.0f);
}

static uint64_t get_time_seed(){
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return ((uint64_t)ts.tv_sec << 32) ^ ts.tv_nsec;
}

static PyObject *manual_seed(PyObject *self, PyObject *args){
  uint64_t seed;
  if(!PyArg_ParseTuple(args,"K",&seed)) return NULL;
  global_seed = seed;
  Py_RETURN_NONE;
}

static PyObject *rand_(PyObject *self, PyObject *args){
  PyObject *shape_obj;
  PyObject *seed_obj;
  if(!PyArg_ParseTuple(args, "OO", &shape_obj, &seed_obj)) return NULL;
  if(!PyTuple_Check(shape_obj)){
    PyErr_SetString(PyExc_TypeError, "shape must be a tuple");
    return NULL;
  }
  int ndim = PyTuple_Size(shape_obj);
  if(ndim <= 0){
    PyErr_SetString(PyExc_ValueError, "shape cannot be empty");
    return NULL;
  }
  uint64_t seed;
  if(seed_obj == Py_None){
    if(global_seed == 0)
      global_seed = get_time_seed();
    seed = global_seed++;
  }else{
    seed = PyLong_AsUnsignedLongLong(seed_obj);
    if(PyErr_Occurred()) return NULL;
  }
  tensor_t *t = (tensor_t*)malloc(sizeof(tensor_t));
  if(!t){
    PyErr_SetString(PyExc_MemoryError,"tensor allocation failed");
    return NULL;
  }
  storage_t *s = (storage_t*)malloc(sizeof(storage_t));
  if(!s){
    free(t);
    PyErr_SetString(PyExc_MemoryError,"storage allocation failed");
    return NULL;
  }
  size_t *shape = (size_t*)malloc(sizeof(size_t)*ndim);
  size_t *stride = (size_t*)malloc(sizeof(size_t)*ndim);
  if(!shape || !stride){
    if(shape) free(shape);
    if(stride) free(stride);
    free(s);
    free(t);
    PyErr_SetString(PyExc_MemoryError,"shape allocation failed");
    return NULL;
  }
  size_t N = 1;
  for(int i=0;i<ndim;i++){
    PyObject *item = PyTuple_GetItem(shape_obj,i);
    size_t dim = PyLong_AsSize_t(item);
    if(PyErr_Occurred()){
      free(shape); free(stride); free(s); free(t);
      return NULL;
    }
    shape[i] = dim;
    N *= dim;
  }
  stride[ndim-1] = 1;
  for(int i=ndim-2;i>=0;i--)
    stride[i] = stride[i+1]*shape[i+1];
  size_t bytes = N*sizeof(float32);
  s->ptr = malloc(bytes);
  if(!s->ptr){
    free(shape); free(stride); free(s); free(t);
    PyErr_SetString(PyExc_MemoryError,"memory allocation failed");
    return NULL;
  }
  s->bytes = bytes;
  s->device.type = CPU;
  s->device.index = 0;
  s->refcount = 1;
  t->storage = s;
  t->buf = s->ptr;
  t->dtype = FP32;
  t->element_size = sizeof(float32);
  t->size = N;
  t->ndim = ndim;
  t->offset = 0;
  t->device.type = CPU;
  t->device.index = 0;
  t->shape = shape;
  t->stride = stride;
  float *out = (float*)s->ptr;
  for(size_t i=0;i<N;i+=4){
    uint32_t ctr[4];
    uint32_t key[2];
    uint64_t counter = i/4;
    ctr[0] = (uint32_t)counter;
    ctr[1] = (uint32_t)(counter>>32);
    ctr[2] = 0;
    ctr[3] = 0;
    key[0] = (uint32_t)seed;
    key[1] = (uint32_t)(seed>>32);
    philox4x32(ctr,key);
    if(i+0<N) out[i+0] = uint32_to_uniform(ctr[0]);
    if(i+1<N) out[i+1] = uint32_to_uniform(ctr[1]);
    if(i+2<N) out[i+2] = uint32_to_uniform(ctr[2]);
    if(i+3<N) out[i+3] = uint32_to_uniform(ctr[3]);
  }
  return PyCapsule_New(t,"tensor_t on CPU",capsule_destroyer);
}

static PyMethodDef methods[] = {
  {"rand", rand_, METH_VARARGS, "random tensor"},
  {"manual_seed", manual_seed, METH_VARARGS, "set global RNG seed"},
  {NULL,NULL,0,NULL}
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "random",
  NULL,
  -1,
  methods
};

PyMODINIT_FUNC PyInit_random(void){
  if(global_seed == 0)
    global_seed = get_time_seed();
  return PyModule_Create(&module);
}
