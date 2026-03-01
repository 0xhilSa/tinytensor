#ifndef TT_MEMORY_H
#define TT_MEMORY_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

size_t tt_available_memory(void);
void* tt_malloc(size_t bytes);
void tt_free(void *ptr);

#ifdef __cplusplus
}
#endif

#endif
