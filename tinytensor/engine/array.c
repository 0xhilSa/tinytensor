#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "dtypes.h"

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

static size_t dtype_size(dtype_t dtype){
  switch(dtype){
    case UINT8: return sizeof(u8);
    case UINT16: return sizeof(u16);
    case UINT32: return sizeof(u32);
    case UINT64: return sizeof(u64);
    case INT8: return sizeof(i8);
    case INT16: return sizeof(i16);
    case INT32: return sizeof(i32);
    case INT64: return sizeof(i64);
    case FP32: return sizeof(f32);
    case FP64: return sizeof(f64);
    case FP128: return sizeof(f128);
    case CMPX64: return sizeof(c64);
    case CMPX128: return sizeof(c128);
    case CMPX256: return sizeof(c256);
    default: return 0;
  }
}

array_t create(size_t ndim, const size_t *shape, dtype_t dtype){
  array_t arr;
  arr.ndim = ndim;
  arr.dtype = dtype;
  arr.elem_size = dtype_size(dtype);
  if(arr.elem_size == 0){
    arr.ndim = 0;
    arr.dtype = 0;
    return arr;
  }
  if(ndim == 0 || arr.elem_size == 0){
    arr.data = NULL;
    arr.shape = NULL;
    arr.length = 0;
    return arr;
  }
  arr.shape = malloc(ndim * sizeof(size_t));
  arr.length = 1;
  for(int i = 0; i < ndim; i++){
    arr.shape[i] = shape[i];
    arr.length *= shape[i];
  }
  arr.data = malloc(arr.length * arr.elem_size);
  if(!arr.data){
    free(arr.shape);
    arr.shape = NULL;
    arr.length = 0;
  }
  return arr;
}

void destroy(array_t *arr){
  if(!arr) return;

  free(arr->data);
  free(arr->shape);

  arr->data = NULL;
  arr->shape = NULL;

  arr->ndim = 0;
  arr->length = 0;
  arr->elem_size = 0;
  arr->dtype = 0;
}
