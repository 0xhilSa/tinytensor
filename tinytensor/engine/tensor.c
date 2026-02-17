// ======================================================
//
// trying to implement the tinytensor engine from scratch
//
// ======================================================

#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
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

typedef enum { CPU, CUDA } DEVICES_t;

typedef struct {
  uint16 bits;
} float16_t;

#define float16 float16_t
#define half float16_t

static inline float16_t float16_from_bits(uint16 raw){
  float16_t h;
  h.bits = raw;
  return h;
}

static inline uint32 float_to_bits(float f){
  uint32 u;
  memcpy(&u, &f, sizeof(u));
  return u;
}

static inline float bits_to_float(uint32 u){
  float f;
  memcpy(&f, &u, sizeof(f));
  return f;
}

float fp16_to_float(float16_t h){
  uint16 x = h.bits;
  uint32 sign = (x >> 15) & 0x1;
  uint32 exp = (x >> 10) & 0x1F;
  uint32 mant = x & 0x3FF;
  uint32 out_sign = sign << 31;
  uint32 out_exp, out_mant;
  if(exp == 0){
    if(mant == 0){
      out_exp = 0;
      out_mant = 0;
    }else{
      exp = 1;
      while((mant & 0x400) == 0){
        mant <<= 1;
        exp--;
      }
      mant &= 0x3FF;
      out_exp  = (exp + (127 - 15)) << 23;
      out_mant = mant << 13;
    }
  }
  else if(exp == 31){
    out_exp  = 255 << 23;
    out_mant = mant << 13;
  }
  else{
    out_exp  = (exp + (127 - 15)) << 23;
    out_mant = mant << 13;
  }
  return bits_to_float(out_sign | out_exp | out_mant);
}

float16_t float_to_fp16(float f){
  uint32 bits = float_to_bits(f);
  uint32 sign = (bits >> 31) & 0x1;
  int32 exp = (bits >> 23) & 0xFF;
  uint32 mant = bits & 0x7FFFFF;
  float16_t out;
  if(exp == 255){
    if(mant == 0){ out.bits = (sign << 15) | (0x1F << 10); }
    else{ out.bits = (sign << 15) | (0x1F << 10) | 0x200; }
    return out;
  }
  int32 new_exp = exp - 127 + 15;
  if(new_exp >= 31){
    out.bits = (sign << 15) | (0x1F << 10);
    return out;
  }
  if(new_exp <= 0){
    if(new_exp < -10){
      out.bits = (sign << 15);
      return out;
    }
    mant |= 0x800000;
    int shift = 1 - new_exp;
    uint32 sub_mant = mant >> (shift + 13);
    uint32 round_bit = (mant >> (shift + 12)) & 1;
    uint32 sticky    = mant & ((1U << (shift + 12)) - 1);
    if(round_bit && (sticky || (sub_mant & 1))){
      sub_mant++;
    }
    out.bits = (sign << 15) | (uint16)sub_mant;
    return out;
  }
  uint32 half_mant = mant >> 13;
  uint32 round_bit = (mant >> 12) & 1;
  uint32 sticky    = mant & 0xFFF;
  if(round_bit && (sticky || (half_mant & 1))){
    half_mant++;
  }
  if(half_mant == 0x400){
    half_mant = 0;
    new_exp++;
    if(new_exp >= 31){
      out.bits = (sign << 15) | (0x1F << 10);
      return out;
    }
  }
  out.bits = (sign << 15) | (new_exp << 10) | (half_mant & 0x3FF);
  return out;
}

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
    case FP16: return sizeof(float16);
    case FP32: return sizeof(float32);
    case FP64: return sizeof(float64);
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
