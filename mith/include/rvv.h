#ifndef _RVV_H
#define _RVV_H

#include <riscv_vector.h>

void* vec_memcpy(void *s1, const void *s2, size_t n);
void* vec_memset(void *s, int c, size_t n);
void* vec_memmove(void *s1, const void *s2, size_t n);
#endif
