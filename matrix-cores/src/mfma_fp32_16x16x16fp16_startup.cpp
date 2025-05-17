/*
Copyright (c) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "helper.hpp"
#include <hip/hip_runtime.h>
#include <iostream>
#include <random>
#include <vector>

/*
This example code uses the mfma intrinsic __builtin_amdgcn_mfma_f32_16x16x16f16
to compute a 16x16x16 matrix multiplication.

Input:
  A : 16 x 16 float16s (a 16x16 matrix)
  B : 16 x 16 float16s (a 16x16 matrix)

Output:
  D : 16 x 16 floats (a 16x16 matrix)
*/

constexpr int M = 16;
constexpr int N = 16;
constexpr int K = 16;

constexpr int LDA = K;
constexpr int LDB = N;
constexpr int LDD = N;

constexpr int A_size = M * LDA;
constexpr int B_size = K * LDB;
constexpr int D_size = M * LDD;

__global__ void sgemm_16x16x16(float *D) {
  using float16x4 =
      __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
  using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

  floatx4 d0 = {0};
  float16x4 a0;
  float16x4 b0;
  a0[0] = 1;
  a0[1] = 2;
  a0[2] = 3;
  a0[3] = 4;
  b0[0] = 5;
  b0[1] = 6;
  b0[2] = 7;
  b0[3] = 8;

  floatx4 d1 = {0};
  float16x4 a1;
  float16x4 b1;
  a1[0] = 9;
  a1[1] = 10;
  a1[2] = 11;
  a1[3] = 12;
  b1[0] = 13;
  b1[1] = 14;
  b1[2] = 15;
  b1[3] = 16;

  floatx4 d2 = {0};
  float16x4 a2;
  float16x4 b2;
  a2[0] = 17;
  a2[1] = 18;
  a2[2] = 19;
  a2[3] = 20;
  b2[0] = 21;
  b2[1] = 22;
  b2[2] = 23;
  b2[3] = 24;

  floatx4 d3 = {0};
  float16x4 a3;
  float16x4 b3;
  a3[0] = 25;
  a3[1] = 26;
  a3[2] = 27;
  a3[3] = 28;
  b3[0] = 29;
  b3[1] = 30;
  b3[2] = 31;
  b3[3] = 32;

  floatx4 d4 = {0};
  float16x4 a4;
  float16x4 b4;
  a4[0] = 33;
  a4[1] = 34;
  a4[2] = 35;
  a4[3] = 36;
  b4[0] = 37;
  b4[1] = 38;
  b4[2] = 39;
  b4[3] = 40;

  floatx4 d5 = {0};
  float16x4 a5;
  float16x4 b5;
  a5[0] = 41;
  a5[1] = 42;
  a5[2] = 43;
  a5[3] = 44;
  b5[0] = 45;
  b5[1] = 46;
  b5[2] = 47;
  b5[3] = 48;

  floatx4 d6 = {0};
  float16x4 a6;
  float16x4 b6;
  a6[0] = 49;
  a6[1] = 50;
  a6[2] = 51;
  a6[3] = 52;
  b6[0] = 53;
  b6[1] = 54;
  b6[2] = 55;
  b6[3] = 56;

  floatx4 d7 = {0};
  float16x4 a7;
  float16x4 b7;
  a7[0] = 57;
  a7[1] = 58;
  a7[2] = 59;
  a7[3] = 60;
  b7[0] = 61;
  b7[1] = 62;
  b7[2] = 63;
  b7[3] = 64;

  d0 = __builtin_amdgcn_mfma_f32_16x16x16f16(a0, b0, d0, 0, 0, 0);
  d1 = __builtin_amdgcn_mfma_f32_16x16x16f16(a1, b1, d1, 0, 0, 0);
  d2 = __builtin_amdgcn_mfma_f32_16x16x16f16(a2, b2, d2, 0, 0, 0);
  d3 = __builtin_amdgcn_mfma_f32_16x16x16f16(a3, b3, d3, 0, 0, 0);
  d4 = __builtin_amdgcn_mfma_f32_16x16x16f16(a4, b4, d4, 0, 0, 0);
  d5 = __builtin_amdgcn_mfma_f32_16x16x16f16(a5, b5, d5, 0, 0, 0);
  d6 = __builtin_amdgcn_mfma_f32_16x16x16f16(a6, b6, d6, 0, 0, 0);
  d7 = __builtin_amdgcn_mfma_f32_16x16x16f16(a7, b7, d7, 0, 0, 0);

  for (int i = 0; i < 4; ++i) {
    const int d_idx =
        threadIdx.x // consecutive threads cover 16 consecutive columns
        + i * LDD   // consecutive registers take consecutive rows of 16 floats
        + threadIdx.y * 4 * LDD; // groups of 16 lanes skip 4 rows

    D[d_idx] = d0[i] + d1[i] + d2[i] + d3[i] + d4[i] + d5[i] + d6[i] + d7[i];
  }
}

int main() {
  // Make and populate device buffers
  float *D_d;
  HIP_CHECK(hipMalloc(&D_d, D_size * sizeof(float)));

  // Launch GEMM kernel
  sgemm_16x16x16<<<304, dim3(16, 16)>>>(D_d);
  HIP_CHECK(hipGetLastError());

  // Copy result back to host
  std::vector<float> D_h(D_size);
  HIP_CHECK(hipMemcpy(D_h.data(), D_d, D_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(D_d));
  return 0;
}
