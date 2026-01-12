#include <stdio.h>
#include <string.h>
#include "ten.h"

int main(){
  int src[] = {1,2,3,4,5,6,7,8,9,10,11,12};
  size_t shape[] = {4,3};
  size_t ndim = 2;
  array_t arr = create(ndim, shape, INT32);
  memcpy(arr.data, src, arr.length * arr.elem_size);
  printf("arr was created then destroyed\n");
  destroy(&arr);
}
