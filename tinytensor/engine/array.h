#ifndef ARRAY_H
#define ARRAY_H

#include <stddef.h>

typedef enum {
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

typedef struct {
  void *data;
  size_t ndim;
  size_t *shape;
  size_t length;
  size_t elem_size;
  dtype_t dtype;
} array_t;

array_t create(size_t ndim, const size_t *shape, dtype_t dtype);
void destroy(array_t *arr);

#endif
