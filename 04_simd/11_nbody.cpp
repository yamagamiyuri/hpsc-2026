#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  __m512 xj_vec = _mm512_load_ps(x);
  __m512 yj_vec = _mm512_load_ps(y);
  __m512 mj_vec = _mm512_load_ps(m);
  
  for(int i=0; i<N; i++) {
    __m512 xi_vec = _mm512_set1_ps(x[i]);
    __m512 yi_vec = _mm512_set1_ps(y[i]);

    __m512 rx_vec = _mm512_sub_ps(xi_vec, xj_vec);
    __m512 ry_vec = _mm512_sub_ps(yi_vec, yj_vec);

    __m512 rx2_vec = _mm512_mul_ps(rx_vec, rx_vec);
    __m512 ry2_vec = _mm512_mul_ps(ry_vec, ry_vec);
    __m512 r2_vec = _mm512_add_ps(rx2_vec, ry2_vec);
    
    __m512 inv_r_vec = _mm512_rsqrt14_ps(r2_vec);

    __m512 inv_r3_vec = _mm512_mul_ps(inv_r_vec, _mm512_mul_ps(inv_r_vec, inv_r_vec));

    __m512 m_inv_r3_vec = _mm512_mul_ps(mj_vec, inv_r3_vec);
    __m512 dfx_vec = _mm512_mul_ps(rx_vec, m_inv_r3_vec);
    __m512 dfy_vec = _mm512_mul_ps(ry_vec, m_inv_r3_vec);

    __mmask16 mask = ~(1 << i);
    __m512 zero = _mm512_setzero_ps();
    dfx_vec = _mm512_mask_blend_ps(mask, zero, dfx_vec);
    dfy_vec = _mm512_mask_blend_ps(mask, zero, dfy_vec);

    fx[i] -= _mm512_reduce_add_ps(dfx_vec); [cite: 573]
    fy[i] -= _mm512_reduce_add_ps(dfy_vec); [cite: 573]
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
