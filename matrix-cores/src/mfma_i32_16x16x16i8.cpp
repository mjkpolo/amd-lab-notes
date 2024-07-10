/*
Copyright (c) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstddef>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include "helper.hpp"

/*
This example code uses the mfma intrinsic __builtin_amdgcn_mfma_i32_16x16x16i8 to
compute a 16x16x16 matrix multiplication.

Input:
  A : 16 x 16 int8s (a 16x16 matrix)
  B : 16 x 16 int8s (a 16x16 matrix)

Output:
  D : 16 x 16 int32s (a 16x16 matrix)
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


__global__ void igemm_16x16x16(const int8_t* A, const int8_t* B, int32_t* D, size_t* cycles)
{

#if __gfx90a__ || __gfx908__
  using int32x4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
  // This kernel computes a 16x16x16 matrix multiplication using a single wavefront.
  int32x4 d = {0}; // zero out 4 vanilla VGPRs

  /*
  One invocation of v_mfma_i32_16x16x16i8 accumulates the sum of 16 outer products,
  16 columns of A with 16 rows of B, into result matrix D (which is in AccVGPRs).
  So we need only one iteration to compute the full matrix D

  For both the 16 columns of A, and the 16 rows of B, we use a single VGPR.
  With 64 lanes, and 4 int8 values per lane, that covers the 16 columns of A and 16
  rows of B.
  Matrix A is a 16 x 16 matrix that is stored in 1 VGPR as follows:
    a[0] covers columns 0, 4, 8, and 12
    a[1] covers columns 1, 5, 9, and 13
    a[2] covers columns 2, 6, 10, and 14
    a[3] covers columns 3, 7, 11, and 15
    first 16 lanes of a[0] cover column 0 -  last 16 lanes of a[0] cover column 12
    first 16 lanes of a[1] cover column 1 -  last 16 lanes of a[1] cover column 13
    first 16 lanes of a[2] cover column 2 -  last 16 lanes of a[2] cover column 14
    first 16 lanes of a[3] cover column 3 -  last 16 lanes of a[3] cover column 15
  Matrix B is a 16 x 16 matrix that is stored in 1 VGPR as follows:
    b[0] covers rows 0, 4, 8, and 12
    b[1] covers rows 1, 5, 9, and 13
    b[2] covers rows 2, 6, 10, and 14
    b[3] covers rows 3, 7, 11, and 15
    first 16 lanes of b[0] cover row 0 -  last 16 lanes of b[0] cover row 12
    first 16 lanes of b[1] cover row 1 -  last 16 lanes of b[1] cover row 13
    first 16 lanes of b[2] cover row 2 -  last 16 lanes of b[2] cover row 14
    first 16 lanes of b[3] cover row 3 -  last 16 lanes of b[3] cover row 15
  Note that A and B are in row-major order.

  This kernel is called with a single wavefront in dim3(16, 4) layout
  */
  size_t start, end;
  size_t &total = cycles[threadIdx.x+threadIdx.y*16];
  total = 0;

  int8_t a[4];
  int8_t b[4];
#pragma unroll 1
  for(int i = 0; i < 4; ++i){
    const int a_idx =  threadIdx.x * LDA      // consecutive threads cover 16 consecutive rows
                     + i                      // consecutive registers take consecutive columns
                     + threadIdx.y * 4;       // groups of 16 lanes skip 4 columns
    a[i] = A[a_idx];

    const int b_idx =  threadIdx.x            // consecutive threads cover 16 consecutive columns
                     + i * LDB                // consecutive registers take consecutive rows
                     + threadIdx.y * LDB * 4; // groups of 16 lanes skip 4 rows
    b[i] = B[b_idx];
  }

  // d = __builtin_amdgcn_mfma_i32_16x16x16i8(*reinterpret_cast<int32_t*>(a), *reinterpret_cast<int32_t*>(b), d, 0, 0, 0);

  asm volatile("s_waitcnt lgkmcnt(0) & vmcnt(0)\n\t"
               "s_memtime %[start]\n\t"
               "s_waitcnt lgkmcnt(0)\n\t"
               "v_mfma_i32_16x16x16i8 %[D] %[A] %[B] %[C]\n\t"
               "s_memtime %[end]\n\t"
               "s_waitcnt lgkmcnt(0)\n\t"
               : [start] "=r"(start), [end] "=r"(end), [D] "=v"(d)
               : [A] "v"(*reinterpret_cast<int32_t*>(a)), [B] "v"(*reinterpret_cast<int32_t*>(b)), [C] "v"(d)); // just change "v" to "a"
  //                                        ^  ^  ^
  //D(=C)                                   |  |  C(=D)
  //                      16 columns of A---|  |--- 16 rows of B

  /*
  Matrix D is a 16 x 16 matrix that is stored in 4 AccVGPRs as follows:
    d[0] covers rows 0, 4, 8, and 12
    d[1] covers rows 1, 5, 9, and 13
    d[2] covers rows 2, 6, 10, and 14
    d[3] covers rows 3, 7, 11, and 15
    first 16 lanes of d[0] cover row 0 -  last 16 lanes of d[0] cover row 12
    first 16 lanes of d[1] cover row 1 -  last 16 lanes of d[1] cover row 13
    first 16 lanes of d[2] cover row 2 -  last 16 lanes of d[2] cover row 14
    first 16 lanes of d[3] cover row 3 -  last 16 lanes of d[3] cover row 15
  */
  total += end - start;

#pragma unroll 1
  for(int i = 0; i < 4; ++i){
    const int d_idx =  threadIdx.x            // consecutive threads cover 16 consecutive columns
                     + i * LDD                // consecutive registers take consecutive rows of 16 floats
                     + threadIdx.y * 4 * LDD; // groups of 16 lanes skip 4 rows

    D[d_idx] = d[i];
  }
#endif
}


int main(){
  if (!gpuArchCheck("gfx90a") && !gpuArchCheck("gfx908")) {
    std::cout << "mfma_f32_16x16x16f16 instruction only available on gfx908 or later."
              << std::endl;
    exit(-1);
  }

  std::mt19937 gen(0);
  std::uniform_int_distribution<int8_t> dist(-100, 100);

  // Make and populate some host matrices
  std::vector<int8_t> A_h(A_size);
  for(int i = 0; i < A_h.size(); ++i){
    A_h[i] = static_cast<int8_t>(dist(gen));
  }
  std::vector<int8_t> B_h(B_size);
  for(int i = 0; i < B_h.size(); ++i){
    B_h[i] = static_cast<int8_t>(dist(gen));
  }

  // Calculate reference D on host
  std::vector<int32_t> Dref_h(D_size);
  gemm_host(A_h, B_h, Dref_h, M, N, K, LDA, LDB, LDD);

  // Make and populate device buffers
  int8_t *A_d, *B_d;
  size_t *cycles_d, *cycles = new size_t[16*4];
  int32_t *D_d;
  HIP_CHECK(hipMalloc(&A_d, A_size * sizeof(int8_t)));
  HIP_CHECK(hipMalloc(&B_d, B_size * sizeof(int8_t)));
  HIP_CHECK(hipMalloc(&D_d, D_size * sizeof(int32_t)));
  HIP_CHECK(hipMalloc(&cycles_d, 16 * 4 * sizeof(size_t)));
  HIP_CHECK(hipMemcpy(A_d, A_h.data(), A_size * sizeof(int8_t), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h.data(), B_size * sizeof(int8_t), hipMemcpyHostToDevice));

  // Launch GEMM kernel
  igemm_16x16x16<<<1, dim3(16, 4)>>>(A_d, B_d, D_d, cycles_d);
  HIP_CHECK(hipGetLastError());

  // Copy result back to host
  std::vector<int32_t> D_h(D_size);
  HIP_CHECK(hipMemcpy(D_h.data(), D_d, D_size * sizeof(int32_t), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(cycles, cycles_d, 16 * 4 * sizeof(size_t), hipMemcpyDeviceToHost));

  std::cout << "Sum of squared differences of host/device result matrices: "
            << compute_l2_error(Dref_h, D_h, M, N, LDD, LDD)
            << std::endl;

  for (int i = 0; i < 16 * 4; i++) {
    std::cout << "Cycles[" << i << "]: " << cycles[i] << std::endl;
  }

  delete [] cycles;
  HIP_CHECK(hipFree(D_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(A_d));
  return 0;
}
