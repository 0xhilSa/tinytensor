// ======================================================
//
// trying to implement the tinytensor engine from scratch
//
// ======================================================

#include <cuda_runtime.h>
#include <stdlib.h>
#include "complex.h"

#define bool unsigned char
#define int8 char
#define uint8 unsigned char
#define int16 short
#define uint16 unsigned short
#define int32 int
#define uint32 unsigned int
#define int64 long
#define uint64 unsigned long
#define float32 float
#define float64 double
#define float128 long double
#define complex64 complex64_t
#define complex128 complex128_t

typedef enum { CPU, CUDA } DEVICES_t;

typedef enum {
  ERROR, // error
  BOOL,
  INT8,
  UINT8,
  INT16,
  UINT16,
  INT32,
  UINT32,
  INT64,
  UINT64,
  FP32,
  FP64,
  FP128,
  CMPX64,
  CMPX128,
} dtype_t;

size_t getsize(dtype_t dtype){
  switch(dtype){
    case BOOL: return sizeof(bool);
    case INT8: return sizeof(int8);
    case UINT8: return sizeof(uint8);
    case INT16: return sizeof(int16);
    case UINT16: return sizeof(uint16);
    case INT32: return sizeof(int32);
    case UINT32: return sizeof(uint32);
    case INT64: return sizeof(int64);
    case UINT64: return sizeof(uint64);
    case FP32: return sizeof(float32);
    case FP64: return sizeof(float64);
    case FP128: return sizeof(float128);
    case CMPX64: return sizeof(complex64);
    case CMPX128: return sizeof(complex128);
    default: return 0;
  }
}

typedef struct {
  DEVICES_t type;
  unsigned short index;
} device_t;

typedef struct {
  void *ptr;
  size_t bytes;
  device_t device;
  int refcount;
} storage_t;

typedef struct {
  storage_t *storage;
  void *buf;
  dtype_t dtype;
  size_t element_size;
  size_t size; // length
  size_t ndim;
  size_t *shape;
  size_t *stride;
  device_t device;
} tensor_t;

void destroy(tensor_t *t){
  if(!t) return;
  if(t->shape) free(t->shape);
  if(t->stride) free(t->stride);
  if(t->storage){
    if(--t->storage->refcount == 0){
      if(t->storage->device.type == CUDA) cudaFree(t->storage->ptr);
      else free(t->storage->ptr);
      free(t->storage);
    }
  }
  free(t);
}
