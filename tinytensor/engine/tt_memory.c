#include "tt_memory.h"
#include <stdio.h>
#include <stdlib.h>

void* tt_malloc(size_t bytes){
  void *ptr = malloc(bytes);
  if(!ptr){ fprintf(stderr, "can't allocate memory: you tried to allocate %zu bytes.\n", bytes); }
  return ptr;
}

void tt_free(void *ptr){ free(ptr); }