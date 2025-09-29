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

constexpr int M = 32;
constexpr int N = 32;
constexpr int K = 32;

constexpr int LDA = K;
constexpr int LDB = N;
constexpr int LDD = N;

constexpr int A_size = M * LDA;
constexpr int B_size = K * LDB;
constexpr int D_size = M * LDD;

__global__ void sgemm_16x16x16(const float16_t *A, const float16_t *B,
                               float *D) {

  using float16x4 =
      __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
  using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
  float16x4 a;
  float16x4 b;

  int col1_off = 16;
  int row1_off = 16 * 32;

  // a0,0 @ b0,0 + a0,1 @ b1,0 = d0,0
  // a0,0 @ b0,1 + a0,1 @ b1,1 = d0,1
  // a1,0 @ b0,0 + a1,1 @ b1,0 = d1,0
  // a1,0 @ b0,1 + a1,1 @ b1,1 = d1,1
  for (int d_col = 0; d_col < 2; d_col++) {
    for (int d_row = 0; d_row < 2; d_row++) {
      floatx4 d = {0};
      for (int i = 0; i < 4; ++i) {
        const int a_idx =
            row1_off * d_row + threadIdx.x * LDA + i + threadIdx.y * 4;
        a[i] = A[a_idx];

        const int b_idx =
            col1_off * d_col + threadIdx.x + i * LDB + threadIdx.y * LDB * 4;
        b[i] = B[b_idx];
      }
      d = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, d, 0, 0, 0);

      for (int i = 0; i < 4; ++i) {
        const int a_idx = row1_off * d_row + col1_off + threadIdx.x * LDA + i +
                          threadIdx.y * 4;
        a[i] = A[a_idx];

        const int b_idx = col1_off * d_col + row1_off + threadIdx.x + i * LDB +
                          threadIdx.y * LDB * 4;
        b[i] = B[b_idx];
      }
      d = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, d, 0, 0, 0);

      for (int i = 0; i < 4; ++i) {
        const int d_idx = row1_off * d_row + col1_off * d_col + threadIdx.x +
                          i * LDD + threadIdx.y * 4 * LDD;

        D[d_idx] = d[i];
      }
    }
  }
}

int main() {
  std::mt19937 gen(0);
  std::uniform_real_distribution<float> dist(-1, 1);

  // Make and populate some host matrices
  std::vector<float16_t> A_h(A_size);
  for (int i = 0; i < A_h.size(); ++i) {
    A_h[i] = static_cast<float16_t>(dist(gen));
  }
  std::vector<float16_t> B_h(B_size);
  for (int i = 0; i < B_h.size(); ++i) {
    B_h[i] = static_cast<float16_t>(dist(gen));
  }

  // Calculate reference D on host
  std::vector<float> Dref_h(D_size);
  gemm_host(A_h, B_h, Dref_h, M, N, K, LDA, LDB, LDD);

  // Make and populate device buffers
  float16_t *A_d, *B_d;
  float *D_d;
  HIP_CHECK(hipMalloc(&A_d, A_size * sizeof(float16_t)));
  HIP_CHECK(hipMalloc(&B_d, B_size * sizeof(float16_t)));
  HIP_CHECK(hipMalloc(&D_d, D_size * sizeof(float)));
  HIP_CHECK(hipMemcpy(A_d, A_h.data(), A_size * sizeof(float16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h.data(), B_size * sizeof(float16_t),
                      hipMemcpyHostToDevice));

  // Launch GEMM kernel
  sgemm_16x16x16<<<1, dim3(16, 4)>>>(A_d, B_d, D_d);
  HIP_CHECK(hipGetLastError());

  // Copy result back to host
  std::vector<float> D_h(D_size);
  HIP_CHECK(hipMemcpy(D_h.data(), D_d, D_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  std::cout << "Sum of squared differences of host/device result matrices: "
            << compute_l2_error(Dref_h, D_h, M, N, LDD, LDD) << std::endl;

  HIP_CHECK(hipFree(D_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(A_d));
  return 0;
}
