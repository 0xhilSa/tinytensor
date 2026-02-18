#ifndef FP_H
#define FP_H

#include <string.h>

#ifdef __CUDACC__
  #define TT_HD __host__ __device__
#else
  #define TT_HD
#endif

typedef struct {
  unsigned short bits;
} float16_t;

#define float16 float16_t
#define half float16_t

TT_HD static inline float16_t float16_from_bits(unsigned short raw){
  float16_t h;
  h.bits = raw;
  return h;
}

TT_HD static inline unsigned int float_to_bits(float f){
  unsigned int u;
  memcpy(&u, &f, sizeof(u));
  return u;
}

TT_HD static inline float bits_to_float(unsigned int u){
  float f;
  memcpy(&f, &u, sizeof(f));
  return f;
}

TT_HD static inline float fp16_to_float(float16_t h){
  unsigned short x = h.bits;
  unsigned int sign = (x >> 15) & 0x1;
  unsigned int exp = (x >> 10) & 0x1F;
  unsigned int mant = x & 0x3FF;
  unsigned int out_sign = sign << 31;
  unsigned int out_exp, out_mant;
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

TT_HD static inline float16_t float_to_fp16(float f){
  unsigned int bits = float_to_bits(f);
  unsigned int sign = (bits >> 31) & 0x1;
  int exp = (bits >> 23) & 0xFF;
  unsigned int mant = bits & 0x7FFFFF;
  float16_t out;
  if(exp == 255){
    if(mant == 0){ out.bits = (sign << 15) | (0x1F << 10); }
    else{ out.bits = (sign << 15) | (0x1F << 10) | 0x200; }
    return out;
  }
  int new_exp = exp - 127 + 15;
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
    unsigned int sub_mant = mant >> (shift + 13);
    unsigned int round_bit = (mant >> (shift + 12)) & 1;
    unsigned int sticky    = mant & ((1U << (shift + 12)) - 1);
    if(round_bit && (sticky || (sub_mant & 1))){
      sub_mant++;
    }
    out.bits = (sign << 15) | (unsigned short)sub_mant;
    return out;
  }
  unsigned int half_mant = mant >> 13;
  unsigned int round_bit = (mant >> 12) & 1;
  unsigned int sticky    = mant & 0xFFF;
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

#endif
