#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include "dtypes.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  BOOL,
  UINT8,
  UINT16,
  UINT32,
  UINT64,
  INT8,
  INT16,
  INT32,
  INT64,
  FP32,
  FP64,
  FP128,
  CMPX64,
  CMPX128,
  CMPX256
} dtype_t;

typedef enum {
  CPU,
  CUDA
} device_t;

typedef struct {
  void *data;
  size_t ndim;
  size_t *shape;
  size_t length;
  size_t elem_size;
  device_t device;
  dtype_t dtype;
} tensor_t;

tensor_t create(size_t ndim, const size_t *shape, device_t device, dtype_t dtype);
void destroy(tensor_t *arr);
void *get(const tensor_t *tensor, const size_t *indices);
void set(const tensor_t *tensor, const size_t *indices, void *value);

#ifdef __cplusplus
}
#endif

#endif
