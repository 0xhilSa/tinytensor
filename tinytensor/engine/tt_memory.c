#include "tt_memory.h"
#include <stdio.h>
#include <stdlib.h>

size_t tt_available_memory(){
  FILE *f = fopen("/proc/meminfo", "r");
  if(!f) return 0;
  char line[256];
  size_t mem = 0;
  while(fgets(line, sizeof(line), f)) {
    if(sscanf(line, "MemAvailable: %zu kB", &mem) == 1) {
      fclose(f);
      return mem * 1024;
    }
  }
  fclose(f);
  return 0;
}

void* tt_malloc(size_t bytes){
  size_t available = tt_available_memory();
  if(available && bytes > available / 2){
    fprintf(stderr, "can't allocate memory: you tried to allocate %zu bytes.\n", bytes);
    return NULL;
  }
  return malloc(bytes);
}

void tt_free(void *ptr){ free(ptr); }
