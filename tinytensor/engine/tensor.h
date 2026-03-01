#ifndef TENSOR_H
#define TENSOR_H

#include <cuda_runtime.h>
#include <stdlib.h>
#include "./fp.h"
#include "./complex.h"

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
#define complex64 complex64_t
#define complex128 complex128_t

#ifdef __cplusplus
extern "C" {
#endif

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
  FP16,
  FP32,
  FP64,
  CMPX64,
  CMPX128
} dtype_t;

dtype_t getdtype(char fmt);
size_t getsize(dtype_t dtype);

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

void destroy(tensor_t *tensor);

#ifdef __cplusplus
}
#endif

#endif
