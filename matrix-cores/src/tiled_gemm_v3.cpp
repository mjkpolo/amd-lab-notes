/*
Copyright (c) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
Copyright (c) 2025 University of Central Florida
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
#include <cassert>
#include <hip/hip_runtime.h>
#include <iostream>
#include <random>
#include <vector>

constexpr int inst_size = 16;
constexpr int N = 4 << 10;
constexpr int phases = N / inst_size;
constexpr int n_x_wavefronts = 1;
constexpr int n_y_wavefronts = 1;

constexpr int A_size = N * N;
constexpr int B_size = N * N;
constexpr int D_size = N * N;
constexpr int col_per_cu = 4;

__global__ void sgemm_16x16x16(const float16_t *A, const float16_t *B,
                               float *D) {

  using float16x4 =
      __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
  using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
  float16x4 a;
  float16x4 b;
  // const int row_per_thread = 1;
  floatx4 d[col_per_cu] = {0};

  const int col_off = inst_size;
  const int row_off = inst_size * N;

  const int tid = threadIdx.x + threadIdx.y * blockDim.x;
  const int lane_id = tid % 64;
  const int tid_x = lane_id % 16;
  const int tid_y = lane_id / 16;
  const int wavefront_id = tid / 64;

  const int d_col =
      (blockIdx.x * n_x_wavefronts + wavefront_id % n_x_wavefronts) *
      col_per_cu;
  const int d_row = blockIdx.y * n_y_wavefronts +
                    (wavefront_id / n_x_wavefronts) % n_y_wavefronts;

#pragma unroll
  for (int phase = 0; phase < phases; phase++) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int a_idx =
          row_off * d_row + col_off * phase + tid_x * N + i + tid_y * 4;
      a[i] = A[a_idx];
    }

#pragma unroll
    for (int cu_col = 0; cu_col < col_per_cu; cu_col++) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int b_idx = col_off * (d_col + cu_col) + row_off * phase + tid_x +
                          i * N + tid_y * N * 4;
        b[i] = B[b_idx];
      }
      d[cu_col] =
          __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, d[cu_col], 0, 0, 0);
    }
  }

#pragma unroll
  for (int cu_col = 0; cu_col < col_per_cu; cu_col++) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d_idx = row_off * d_row + col_off * (d_col + cu_col) + tid_x +
                        i * N + tid_y * 4 * N;

      D[d_idx] = d[cu_col][i];
    }
  }
}

int main() {
  std::mt19937 gen(0);
  std::uniform_real_distribution<float> dist(-1, 1);

  assert(N % inst_size == 0);

  std::cout << "Generating A matrix..." << std::endl;
  std::vector<float16_t> A_h(A_size);
  for (int i = 0; i < A_h.size(); ++i) {
    A_h[i] = static_cast<float16_t>(dist(gen));
  }
  std::cout << "Generating B matrix..." << std::endl;
  std::vector<float16_t> B_h(B_size);
  for (int i = 0; i < B_h.size(); ++i) {
    B_h[i] = static_cast<float16_t>(dist(gen));
  }

  std::cout << "Calculating on host..." << std::endl;
  std::vector<float> Dref_h(D_size);
  gemm_host(A_h, B_h, Dref_h, N, N, N, N, N, N);

  std::cout << "Allocating GPU buffers..." << std::endl;
  float16_t *A_d, *B_d;
  float *D_d;
  HIP_CHECK(hipMalloc(&A_d, A_size * sizeof(float16_t)));
  HIP_CHECK(hipMalloc(&B_d, B_size * sizeof(float16_t)));
  HIP_CHECK(hipMalloc(&D_d, D_size * sizeof(float)));
  HIP_CHECK(hipMemcpy(A_d, A_h.data(), A_size * sizeof(float16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h.data(), B_size * sizeof(float16_t),
                      hipMemcpyHostToDevice));

  std::cout << "Launching GPU kernel..." << std::endl;
  sgemm_16x16x16<<<dim3(phases / n_x_wavefronts / col_per_cu,
                        phases / n_y_wavefronts),
                   64 * n_x_wavefronts * n_y_wavefronts>>>(A_d, B_d, D_d);
  HIP_CHECK(hipGetLastError());

  std::cout << "Copying result from GPU..." << std::endl;
  std::vector<float> D_h(D_size);
  HIP_CHECK(hipMemcpy(D_h.data(), D_d, D_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  std::cout << "Sum of squared differences of host/device result matrices: "
            << compute_l2_error(Dref_h, D_h, N, N, N, N) << std::endl;

  HIP_CHECK(hipFree(D_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(A_d));
  return 0;
}
