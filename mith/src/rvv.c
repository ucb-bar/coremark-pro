#include "rvv.h"

void* vec_memcpy(void *s1, const void *s2, size_t n) {
  unsigned char* dst = (unsigned char*)s1;
  const unsigned char* src = (const unsigned char*)s2;
  for (size_t i = 0; i < n;) {
    long vl = __riscv_vsetvl_e8m8(n - i);
    __riscv_vse8_v_u8m8(&dst[i], __riscv_vle8_v_u8m8(&src[i], vl), vl);
    i += vl;
  }
  return s1;
}

void* vec_memset(void *s, int c, size_t n) {
  signed char* dst = (signed char*)s;
  long vl = __riscv_vsetvlmax_e8m8();
  vint8m8_t vr = __riscv_vmv_v_x_i8m8(c, vl);
  for (size_t i = 0; i < n;) {
    long vl = __riscv_vsetvl_e8m8(n - i);
    __riscv_vse8_v_i8m8(&dst[i], vr, vl);
    i += vl;
  }
  return s;
}

void* vec_memmove(void *s1, const void *s2, size_t n) {
  if (s2 < s1 && ((s1 - s2) < n)) {
    return memmove(s1, s2, n);
  } else {
    return vec_memcpy(s1, s2, n);
  }
}
