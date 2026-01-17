#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdint.h>
#include "dtypes.h"

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

typedef struct {
  void *data;
  size_t ndim;
  size_t *shape;
  size_t length;
  size_t elem_size;
  dtype_t dtype;
} tensor_t;

static size_t dtype_size(dtype_t dtype){
  switch(dtype){
    case BOOL: return sizeof(bool);
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

tensor_t create(size_t ndim, const size_t *shape, dtype_t dtype){
  tensor_t arr = {0};
  if(ndim == 0 || !shape) return arr;
  arr.elem_size = dtype_size(dtype);
  if(arr.elem_size == 0) return arr;
  arr.ndim = ndim;
  arr.dtype = dtype;
  arr.shape = malloc(ndim * sizeof(size_t));
  if(!arr.shape) return arr;
  arr.length = 1;
  for(size_t i = 0; i < ndim; i++){
    if(shape[i] == 0 ||
       arr.length > SIZE_MAX / shape[i]){
      free(arr.shape);
      arr.shape = NULL;
      arr.length = 0;
      return arr;
    }
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

void destroy(tensor_t *arr){
  if(!arr) return;
  free(arr->data); // WARNING: use when that pointer is at the host/CPU/DISK *CUDA allocated memory won't be free*
  free(arr->shape);
  arr->data = NULL;
  arr->shape = NULL;
  arr->ndim = 0;
  arr->length = 0;
  arr->elem_size = 0;
  arr->dtype = 0;
}

void *get(const tensor_t *tensor, const size_t *indices){
  if(!tensor || !tensor->data || !indices) return NULL;
  size_t linear_index = 0;
  size_t stride = 1;
  for(size_t i = tensor->ndim; i-- > 0;){
    if(indices[i] >= tensor->shape[i]) return NULL;
    linear_index += indices[i] * stride;
    stride *= tensor->shape[i];
  }
  return (char *)tensor->data + linear_index * tensor->elem_size;
}

void set(tensor_t *tensor, const size_t *indices, void *value){
  if(!tensor || !tensor->data || !indices) return;
  size_t linear_index = 0;
  size_t stride = 1;
  for(size_t i = tensor->ndim; i-- >0;){
    if(indices[i] >= tensor->shape[i]) return;
    linear_index += indices[i] * stride;
    stride *= tensor->shape[i];
  }
  memcpy(
    (char *)tensor->data + linear_index * tensor->elem_size,
    value,
    tensor->elem_size
  );
}
