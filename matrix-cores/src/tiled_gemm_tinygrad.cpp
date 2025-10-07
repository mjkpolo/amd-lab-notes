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

constexpr int A_size = N * N;
constexpr int B_size = N * N;
constexpr int D_size = N * N;

#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
typedef long unsigned int size_t;
#define half _Float16
typedef half half4 __attribute__((ext_vector_type(4)));
static inline __attribute__((device)) half4 make_half4(half x, half y, half z,
                                                       half w) {
  return {x, y, z, w};
}
typedef float float4_ __attribute__((ext_vector_type(4)));
static inline __attribute__((device)) float4_ make_float4_(float x, float y,
                                                           float z, float w) {
  return {x, y, z, w};
}
#define __WMMA_16_16_16_half_float __builtin_amdgcn_mfma_f32_16x16x16f16
extern "C" __attribute__((global)) void
    __attribute__((amdgpu_flat_work_group_size(1, 512)))
    r_64_4_64_4_2_2_2_4_4_2_256(float *data0_16777216, half *data1_16777216,
                                half *data2_16777216) {
  float acc0[128];
  int gidx0 = blockIdx.x;  /* 4 */
  int gidx1 = blockIdx.y;  /* 64 */
  int lidx0 = threadIdx.x; /* 64 */
  int lidx1 = threadIdx.y; /* 4 */
  int lidx2 = threadIdx.z; /* 2 */
  int alu0 = (lidx0 & 1);
  int alu1 = ((lidx0 >> 1) & 1);
  int alu2 = ((lidx0 >> 2) & 1);
  int alu3 = ((lidx0 >> 3) & 1);
  int alu4 = ((lidx0 >> 4) & 1);
  int alu5 = (lidx0 >> 5);
  int alu6 = (gidx1 << 18);
  int alu7 = ((gidx0 << 10) + (lidx2 << 9) + (lidx1 << 7) + (alu3 << 3) +
              (alu2 << 2) + (alu1 << 1) + alu0);
  int alu8 = (alu4 << 14);
  int alu9 = (alu5 << 15);
  *(acc0 + 0) = 0.0f;
  *(acc0 + 1) = 0.0f;
  *(acc0 + 2) = 0.0f;
  *(acc0 + 3) = 0.0f;
  *(acc0 + 4) = 0.0f;
  *(acc0 + 5) = 0.0f;
  *(acc0 + 6) = 0.0f;
  *(acc0 + 7) = 0.0f;
  *(acc0 + 8) = 0.0f;
  *(acc0 + 9) = 0.0f;
  *(acc0 + 10) = 0.0f;
  *(acc0 + 11) = 0.0f;
  *(acc0 + 12) = 0.0f;
  *(acc0 + 13) = 0.0f;
  *(acc0 + 14) = 0.0f;
  *(acc0 + 15) = 0.0f;
  *(acc0 + 16) = 0.0f;
  *(acc0 + 17) = 0.0f;
  *(acc0 + 18) = 0.0f;
  *(acc0 + 19) = 0.0f;
  *(acc0 + 20) = 0.0f;
  *(acc0 + 21) = 0.0f;
  *(acc0 + 22) = 0.0f;
  *(acc0 + 23) = 0.0f;
  *(acc0 + 24) = 0.0f;
  *(acc0 + 25) = 0.0f;
  *(acc0 + 26) = 0.0f;
  *(acc0 + 27) = 0.0f;
  *(acc0 + 28) = 0.0f;
  *(acc0 + 29) = 0.0f;
  *(acc0 + 30) = 0.0f;
  *(acc0 + 31) = 0.0f;
  *(acc0 + 32) = 0.0f;
  *(acc0 + 33) = 0.0f;
  *(acc0 + 34) = 0.0f;
  *(acc0 + 35) = 0.0f;
  *(acc0 + 36) = 0.0f;
  *(acc0 + 37) = 0.0f;
  *(acc0 + 38) = 0.0f;
  *(acc0 + 39) = 0.0f;
  *(acc0 + 40) = 0.0f;
  *(acc0 + 41) = 0.0f;
  *(acc0 + 42) = 0.0f;
  *(acc0 + 43) = 0.0f;
  *(acc0 + 44) = 0.0f;
  *(acc0 + 45) = 0.0f;
  *(acc0 + 46) = 0.0f;
  *(acc0 + 47) = 0.0f;
  *(acc0 + 48) = 0.0f;
  *(acc0 + 49) = 0.0f;
  *(acc0 + 50) = 0.0f;
  *(acc0 + 51) = 0.0f;
  *(acc0 + 52) = 0.0f;
  *(acc0 + 53) = 0.0f;
  *(acc0 + 54) = 0.0f;
  *(acc0 + 55) = 0.0f;
  *(acc0 + 56) = 0.0f;
  *(acc0 + 57) = 0.0f;
  *(acc0 + 58) = 0.0f;
  *(acc0 + 59) = 0.0f;
  *(acc0 + 60) = 0.0f;
  *(acc0 + 61) = 0.0f;
  *(acc0 + 62) = 0.0f;
  *(acc0 + 63) = 0.0f;
  *(acc0 + 64) = 0.0f;
  *(acc0 + 65) = 0.0f;
  *(acc0 + 66) = 0.0f;
  *(acc0 + 67) = 0.0f;
  *(acc0 + 68) = 0.0f;
  *(acc0 + 69) = 0.0f;
  *(acc0 + 70) = 0.0f;
  *(acc0 + 71) = 0.0f;
  *(acc0 + 72) = 0.0f;
  *(acc0 + 73) = 0.0f;
  *(acc0 + 74) = 0.0f;
  *(acc0 + 75) = 0.0f;
  *(acc0 + 76) = 0.0f;
  *(acc0 + 77) = 0.0f;
  *(acc0 + 78) = 0.0f;
  *(acc0 + 79) = 0.0f;
  *(acc0 + 80) = 0.0f;
  *(acc0 + 81) = 0.0f;
  *(acc0 + 82) = 0.0f;
  *(acc0 + 83) = 0.0f;
  *(acc0 + 84) = 0.0f;
  *(acc0 + 85) = 0.0f;
  *(acc0 + 86) = 0.0f;
  *(acc0 + 87) = 0.0f;
  *(acc0 + 88) = 0.0f;
  *(acc0 + 89) = 0.0f;
  *(acc0 + 90) = 0.0f;
  *(acc0 + 91) = 0.0f;
  *(acc0 + 92) = 0.0f;
  *(acc0 + 93) = 0.0f;
  *(acc0 + 94) = 0.0f;
  *(acc0 + 95) = 0.0f;
  *(acc0 + 96) = 0.0f;
  *(acc0 + 97) = 0.0f;
  *(acc0 + 98) = 0.0f;
  *(acc0 + 99) = 0.0f;
  *(acc0 + 100) = 0.0f;
  *(acc0 + 101) = 0.0f;
  *(acc0 + 102) = 0.0f;
  *(acc0 + 103) = 0.0f;
  *(acc0 + 104) = 0.0f;
  *(acc0 + 105) = 0.0f;
  *(acc0 + 106) = 0.0f;
  *(acc0 + 107) = 0.0f;
  *(acc0 + 108) = 0.0f;
  *(acc0 + 109) = 0.0f;
  *(acc0 + 110) = 0.0f;
  *(acc0 + 111) = 0.0f;
  *(acc0 + 112) = 0.0f;
  *(acc0 + 113) = 0.0f;
  *(acc0 + 114) = 0.0f;
  *(acc0 + 115) = 0.0f;
  *(acc0 + 116) = 0.0f;
  *(acc0 + 117) = 0.0f;
  *(acc0 + 118) = 0.0f;
  *(acc0 + 119) = 0.0f;
  *(acc0 + 120) = 0.0f;
  *(acc0 + 121) = 0.0f;
  *(acc0 + 122) = 0.0f;
  *(acc0 + 123) = 0.0f;
  *(acc0 + 124) = 0.0f;
  *(acc0 + 125) = 0.0f;
  *(acc0 + 126) = 0.0f;
  *(acc0 + 127) = 0.0f;
  for (int ridx1002 = 0; ridx1002 < 256; ridx1002++) {
    int alu138 = (alu6 + (alu3 << 15) + (alu2 << 14) + (alu1 << 13) +
                  (alu0 << 12) + (alu5 << 3) + (ridx1002 << 4) + (alu4 << 2));
    half4 val0 = (*((half4 *)((data1_16777216 + alu138))));
    half4 val1 = (*((half4 *)((data1_16777216 + (alu138 + 65536)))));
    half4 val2 = (*((half4 *)((data1_16777216 + (alu138 + 131072)))));
    half4 val3 = (*((half4 *)((data1_16777216 + (alu138 + 196608)))));
    int alu139 = (alu7 + alu9 + (ridx1002 << 16) + alu8);
    half val4 = (*(data2_16777216 + alu139));
    half val5 = (*(data2_16777216 + (alu139 + 16)));
    half val6 = (*(data2_16777216 + (alu139 + 32)));
    half val7 = (*(data2_16777216 + (alu139 + 48)));
    half val8 = (*(data2_16777216 + (alu139 + 64)));
    half val9 = (*(data2_16777216 + (alu139 + 80)));
    half val10 = (*(data2_16777216 + (alu139 + 96)));
    half val11 = (*(data2_16777216 + (alu139 + 112)));
    half val12 = (*(data2_16777216 + (alu139 + 4096)));
    half val13 = (*(data2_16777216 + (alu139 + 4112)));
    half val14 = (*(data2_16777216 + (alu139 + 4128)));
    half val15 = (*(data2_16777216 + (alu139 + 4144)));
    half val16 = (*(data2_16777216 + (alu139 + 4160)));
    half val17 = (*(data2_16777216 + (alu139 + 4176)));
    half val18 = (*(data2_16777216 + (alu139 + 4192)));
    half val19 = (*(data2_16777216 + (alu139 + 4208)));
    half val20 = (*(data2_16777216 + (alu139 + 8192)));
    half val21 = (*(data2_16777216 + (alu139 + 8208)));
    half val22 = (*(data2_16777216 + (alu139 + 8224)));
    half val23 = (*(data2_16777216 + (alu139 + 8240)));
    half val24 = (*(data2_16777216 + (alu139 + 8256)));
    half val25 = (*(data2_16777216 + (alu139 + 8272)));
    half val26 = (*(data2_16777216 + (alu139 + 8288)));
    half val27 = (*(data2_16777216 + (alu139 + 8304)));
    half val28 = (*(data2_16777216 + (alu139 + 12288)));
    half val29 = (*(data2_16777216 + (alu139 + 12304)));
    half val30 = (*(data2_16777216 + (alu139 + 12320)));
    half val31 = (*(data2_16777216 + (alu139 + 12336)));
    half val32 = (*(data2_16777216 + (alu139 + 12352)));
    half val33 = (*(data2_16777216 + (alu139 + 12368)));
    half val34 = (*(data2_16777216 + (alu139 + 12384)));
    half val35 = (*(data2_16777216 + (alu139 + 12400)));
    half4 cast0 = make_half4(val5, val13, val21, val29);
    half4 cast1 = make_half4(val6, val14, val22, val30);
    half4 cast2 = make_half4(val7, val15, val23, val31);
    half4 cast3 = make_half4(val8, val16, val24, val32);
    half4 cast4 = make_half4(val9, val17, val25, val33);
    half4 cast5 = make_half4(val10, val18, val26, val34);
    half4 cast6 = make_half4(val11, val19, val27, val35);
    half4 cast7 = make_half4(val4, val12, val20, val28);
    // 32 mfmas called
    float4_ wmma0 =
        __WMMA_16_16_16_half_float(val1, cast0,
                                   make_float4_((*(acc0 + 40)), (*(acc0 + 41)),
                                                (*(acc0 + 42)), (*(acc0 + 43))),
                                   0, 0, 0);
    *(acc0 + 40) = wmma0.x;
    *(acc0 + 41) = wmma0.y;
    *(acc0 + 42) = wmma0.z;
    *(acc0 + 43) = wmma0.w;
    float4_ wmma1 =
        __WMMA_16_16_16_half_float(val1, cast1,
                                   make_float4_((*(acc0 + 48)), (*(acc0 + 49)),
                                                (*(acc0 + 50)), (*(acc0 + 51))),
                                   0, 0, 0);
    *(acc0 + 48) = wmma1.x;
    *(acc0 + 49) = wmma1.y;
    *(acc0 + 50) = wmma1.z;
    *(acc0 + 51) = wmma1.w;
    float4_ wmma2 =
        __WMMA_16_16_16_half_float(val1, cast2,
                                   make_float4_((*(acc0 + 56)), (*(acc0 + 57)),
                                                (*(acc0 + 58)), (*(acc0 + 59))),
                                   0, 0, 0);
    *(acc0 + 56) = wmma2.x;
    *(acc0 + 57) = wmma2.y;
    *(acc0 + 58) = wmma2.z;
    *(acc0 + 59) = wmma2.w;
    float4_ wmma3 =
        __WMMA_16_16_16_half_float(val1, cast3,
                                   make_float4_((*(acc0 + 36)), (*(acc0 + 37)),
                                                (*(acc0 + 38)), (*(acc0 + 39))),
                                   0, 0, 0);
    *(acc0 + 36) = wmma3.x;
    *(acc0 + 37) = wmma3.y;
    *(acc0 + 38) = wmma3.z;
    *(acc0 + 39) = wmma3.w;
    float4_ wmma4 =
        __WMMA_16_16_16_half_float(val1, cast4,
                                   make_float4_((*(acc0 + 44)), (*(acc0 + 45)),
                                                (*(acc0 + 46)), (*(acc0 + 47))),
                                   0, 0, 0);
    *(acc0 + 44) = wmma4.x;
    *(acc0 + 45) = wmma4.y;
    *(acc0 + 46) = wmma4.z;
    *(acc0 + 47) = wmma4.w;
    float4_ wmma5 =
        __WMMA_16_16_16_half_float(val1, cast5,
                                   make_float4_((*(acc0 + 52)), (*(acc0 + 53)),
                                                (*(acc0 + 54)), (*(acc0 + 55))),
                                   0, 0, 0);
    *(acc0 + 52) = wmma5.x;
    *(acc0 + 53) = wmma5.y;
    *(acc0 + 54) = wmma5.z;
    *(acc0 + 55) = wmma5.w;
    float4_ wmma6 =
        __WMMA_16_16_16_half_float(val1, cast6,
                                   make_float4_((*(acc0 + 60)), (*(acc0 + 61)),
                                                (*(acc0 + 62)), (*(acc0 + 63))),
                                   0, 0, 0);
    *(acc0 + 60) = wmma6.x;
    *(acc0 + 61) = wmma6.y;
    *(acc0 + 62) = wmma6.z;
    *(acc0 + 63) = wmma6.w;
    float4_ wmma7 =
        __WMMA_16_16_16_half_float(val1, cast7,
                                   make_float4_((*(acc0 + 32)), (*(acc0 + 33)),
                                                (*(acc0 + 34)), (*(acc0 + 35))),
                                   0, 0, 0);
    *(acc0 + 32) = wmma7.x;
    *(acc0 + 33) = wmma7.y;
    *(acc0 + 34) = wmma7.z;
    *(acc0 + 35) = wmma7.w;
    float4_ wmma8 =
        __WMMA_16_16_16_half_float(val2, cast0,
                                   make_float4_((*(acc0 + 72)), (*(acc0 + 73)),
                                                (*(acc0 + 74)), (*(acc0 + 75))),
                                   0, 0, 0);
    *(acc0 + 72) = wmma8.x;
    *(acc0 + 73) = wmma8.y;
    *(acc0 + 74) = wmma8.z;
    *(acc0 + 75) = wmma8.w;
    float4_ wmma9 =
        __WMMA_16_16_16_half_float(val2, cast1,
                                   make_float4_((*(acc0 + 80)), (*(acc0 + 81)),
                                                (*(acc0 + 82)), (*(acc0 + 83))),
                                   0, 0, 0);
    *(acc0 + 80) = wmma9.x;
    *(acc0 + 81) = wmma9.y;
    *(acc0 + 82) = wmma9.z;
    *(acc0 + 83) = wmma9.w;
    float4_ wmma10 =
        __WMMA_16_16_16_half_float(val2, cast2,
                                   make_float4_((*(acc0 + 88)), (*(acc0 + 89)),
                                                (*(acc0 + 90)), (*(acc0 + 91))),
                                   0, 0, 0);
    *(acc0 + 88) = wmma10.x;
    *(acc0 + 89) = wmma10.y;
    *(acc0 + 90) = wmma10.z;
    *(acc0 + 91) = wmma10.w;
    float4_ wmma11 =
        __WMMA_16_16_16_half_float(val2, cast3,
                                   make_float4_((*(acc0 + 68)), (*(acc0 + 69)),
                                                (*(acc0 + 70)), (*(acc0 + 71))),
                                   0, 0, 0);
    *(acc0 + 68) = wmma11.x;
    *(acc0 + 69) = wmma11.y;
    *(acc0 + 70) = wmma11.z;
    *(acc0 + 71) = wmma11.w;
    float4_ wmma12 =
        __WMMA_16_16_16_half_float(val2, cast4,
                                   make_float4_((*(acc0 + 76)), (*(acc0 + 77)),
                                                (*(acc0 + 78)), (*(acc0 + 79))),
                                   0, 0, 0);
    *(acc0 + 76) = wmma12.x;
    *(acc0 + 77) = wmma12.y;
    *(acc0 + 78) = wmma12.z;
    *(acc0 + 79) = wmma12.w;
    float4_ wmma13 =
        __WMMA_16_16_16_half_float(val2, cast5,
                                   make_float4_((*(acc0 + 84)), (*(acc0 + 85)),
                                                (*(acc0 + 86)), (*(acc0 + 87))),
                                   0, 0, 0);
    *(acc0 + 84) = wmma13.x;
    *(acc0 + 85) = wmma13.y;
    *(acc0 + 86) = wmma13.z;
    *(acc0 + 87) = wmma13.w;
    float4_ wmma14 =
        __WMMA_16_16_16_half_float(val2, cast6,
                                   make_float4_((*(acc0 + 92)), (*(acc0 + 93)),
                                                (*(acc0 + 94)), (*(acc0 + 95))),
                                   0, 0, 0);
    *(acc0 + 92) = wmma14.x;
    *(acc0 + 93) = wmma14.y;
    *(acc0 + 94) = wmma14.z;
    *(acc0 + 95) = wmma14.w;
    float4_ wmma15 =
        __WMMA_16_16_16_half_float(val2, cast7,
                                   make_float4_((*(acc0 + 64)), (*(acc0 + 65)),
                                                (*(acc0 + 66)), (*(acc0 + 67))),
                                   0, 0, 0);
    *(acc0 + 64) = wmma15.x;
    *(acc0 + 65) = wmma15.y;
    *(acc0 + 66) = wmma15.z;
    *(acc0 + 67) = wmma15.w;
    float4_ wmma16 = __WMMA_16_16_16_half_float(
        val3, cast0,
        make_float4_((*(acc0 + 104)), (*(acc0 + 105)), (*(acc0 + 106)),
                     (*(acc0 + 107))),
        0, 0, 0);
    *(acc0 + 104) = wmma16.x;
    *(acc0 + 105) = wmma16.y;
    *(acc0 + 106) = wmma16.z;
    *(acc0 + 107) = wmma16.w;
    float4_ wmma17 = __WMMA_16_16_16_half_float(
        val3, cast1,
        make_float4_((*(acc0 + 112)), (*(acc0 + 113)), (*(acc0 + 114)),
                     (*(acc0 + 115))),
        0, 0, 0);
    *(acc0 + 112) = wmma17.x;
    *(acc0 + 113) = wmma17.y;
    *(acc0 + 114) = wmma17.z;
    *(acc0 + 115) = wmma17.w;
    float4_ wmma18 = __WMMA_16_16_16_half_float(
        val3, cast2,
        make_float4_((*(acc0 + 120)), (*(acc0 + 121)), (*(acc0 + 122)),
                     (*(acc0 + 123))),
        0, 0, 0);
    *(acc0 + 120) = wmma18.x;
    *(acc0 + 121) = wmma18.y;
    *(acc0 + 122) = wmma18.z;
    *(acc0 + 123) = wmma18.w;
    float4_ wmma19 = __WMMA_16_16_16_half_float(
        val3, cast3,
        make_float4_((*(acc0 + 100)), (*(acc0 + 101)), (*(acc0 + 102)),
                     (*(acc0 + 103))),
        0, 0, 0);
    *(acc0 + 100) = wmma19.x;
    *(acc0 + 101) = wmma19.y;
    *(acc0 + 102) = wmma19.z;
    *(acc0 + 103) = wmma19.w;
    float4_ wmma20 = __WMMA_16_16_16_half_float(
        val3, cast4,
        make_float4_((*(acc0 + 108)), (*(acc0 + 109)), (*(acc0 + 110)),
                     (*(acc0 + 111))),
        0, 0, 0);
    *(acc0 + 108) = wmma20.x;
    *(acc0 + 109) = wmma20.y;
    *(acc0 + 110) = wmma20.z;
    *(acc0 + 111) = wmma20.w;
    float4_ wmma21 = __WMMA_16_16_16_half_float(
        val3, cast5,
        make_float4_((*(acc0 + 116)), (*(acc0 + 117)), (*(acc0 + 118)),
                     (*(acc0 + 119))),
        0, 0, 0);
    *(acc0 + 116) = wmma21.x;
    *(acc0 + 117) = wmma21.y;
    *(acc0 + 118) = wmma21.z;
    *(acc0 + 119) = wmma21.w;
    float4_ wmma22 = __WMMA_16_16_16_half_float(
        val3, cast6,
        make_float4_((*(acc0 + 124)), (*(acc0 + 125)), (*(acc0 + 126)),
                     (*(acc0 + 127))),
        0, 0, 0);
    *(acc0 + 124) = wmma22.x;
    *(acc0 + 125) = wmma22.y;
    *(acc0 + 126) = wmma22.z;
    *(acc0 + 127) = wmma22.w;
    float4_ wmma23 =
        __WMMA_16_16_16_half_float(val3, cast7,
                                   make_float4_((*(acc0 + 96)), (*(acc0 + 97)),
                                                (*(acc0 + 98)), (*(acc0 + 99))),
                                   0, 0, 0);
    *(acc0 + 96) = wmma23.x;
    *(acc0 + 97) = wmma23.y;
    *(acc0 + 98) = wmma23.z;
    *(acc0 + 99) = wmma23.w;
    float4_ wmma24 =
        __WMMA_16_16_16_half_float(val0, cast0,
                                   make_float4_((*(acc0 + 8)), (*(acc0 + 9)),
                                                (*(acc0 + 10)), (*(acc0 + 11))),
                                   0, 0, 0);
    *(acc0 + 8) = wmma24.x;
    *(acc0 + 9) = wmma24.y;
    *(acc0 + 10) = wmma24.z;
    *(acc0 + 11) = wmma24.w;
    float4_ wmma25 =
        __WMMA_16_16_16_half_float(val0, cast1,
                                   make_float4_((*(acc0 + 16)), (*(acc0 + 17)),
                                                (*(acc0 + 18)), (*(acc0 + 19))),
                                   0, 0, 0);
    *(acc0 + 16) = wmma25.x;
    *(acc0 + 17) = wmma25.y;
    *(acc0 + 18) = wmma25.z;
    *(acc0 + 19) = wmma25.w;
    float4_ wmma26 =
        __WMMA_16_16_16_half_float(val0, cast2,
                                   make_float4_((*(acc0 + 24)), (*(acc0 + 25)),
                                                (*(acc0 + 26)), (*(acc0 + 27))),
                                   0, 0, 0);
    *(acc0 + 24) = wmma26.x;
    *(acc0 + 25) = wmma26.y;
    *(acc0 + 26) = wmma26.z;
    *(acc0 + 27) = wmma26.w;
    float4_ wmma27 =
        __WMMA_16_16_16_half_float(val0, cast3,
                                   make_float4_((*(acc0 + 4)), (*(acc0 + 5)),
                                                (*(acc0 + 6)), (*(acc0 + 7))),
                                   0, 0, 0);
    *(acc0 + 4) = wmma27.x;
    *(acc0 + 5) = wmma27.y;
    *(acc0 + 6) = wmma27.z;
    *(acc0 + 7) = wmma27.w;
    float4_ wmma28 =
        __WMMA_16_16_16_half_float(val0, cast4,
                                   make_float4_((*(acc0 + 12)), (*(acc0 + 13)),
                                                (*(acc0 + 14)), (*(acc0 + 15))),
                                   0, 0, 0);
    *(acc0 + 12) = wmma28.x;
    *(acc0 + 13) = wmma28.y;
    *(acc0 + 14) = wmma28.z;
    *(acc0 + 15) = wmma28.w;
    float4_ wmma29 =
        __WMMA_16_16_16_half_float(val0, cast5,
                                   make_float4_((*(acc0 + 20)), (*(acc0 + 21)),
                                                (*(acc0 + 22)), (*(acc0 + 23))),
                                   0, 0, 0);
    *(acc0 + 20) = wmma29.x;
    *(acc0 + 21) = wmma29.y;
    *(acc0 + 22) = wmma29.z;
    *(acc0 + 23) = wmma29.w;
    float4_ wmma30 =
        __WMMA_16_16_16_half_float(val0, cast6,
                                   make_float4_((*(acc0 + 28)), (*(acc0 + 29)),
                                                (*(acc0 + 30)), (*(acc0 + 31))),
                                   0, 0, 0);
    *(acc0 + 28) = wmma30.x;
    *(acc0 + 29) = wmma30.y;
    *(acc0 + 30) = wmma30.z;
    *(acc0 + 31) = wmma30.w;
    float4_ wmma31 =
        __WMMA_16_16_16_half_float(val0, cast7,
                                   make_float4_((*(acc0 + 0)), (*(acc0 + 1)),
                                                (*(acc0 + 2)), (*(acc0 + 3))),
                                   0, 0, 0);
    *(acc0 + 0) = wmma31.x;
    *(acc0 + 1) = wmma31.y;
    *(acc0 + 2) = wmma31.z;
    *(acc0 + 3) = wmma31.w;
  }
  int alu269 = (alu7 + alu6 + alu9 + alu8);
  *(data0_16777216 + alu269) = (((*(acc0 + 0))));
  *(data0_16777216 + (alu269 + 16)) = (((*(acc0 + 8))));
  *(data0_16777216 + (alu269 + 32)) = (((*(acc0 + 16))));
  *(data0_16777216 + (alu269 + 48)) = (((*(acc0 + 24))));
  *(data0_16777216 + (alu269 + 64)) = (((*(acc0 + 4))));
  *(data0_16777216 + (alu269 + 80)) = (((*(acc0 + 12))));
  *(data0_16777216 + (alu269 + 96)) = (((*(acc0 + 20))));
  *(data0_16777216 + (alu269 + 112)) = (((*(acc0 + 28))));
  *(data0_16777216 + (alu269 + 4096)) = (((*(acc0 + 1))));
  *(data0_16777216 + (alu269 + 4112)) = (((*(acc0 + 9))));
  *(data0_16777216 + (alu269 + 4128)) = (((*(acc0 + 17))));
  *(data0_16777216 + (alu269 + 4144)) = (((*(acc0 + 25))));
  *(data0_16777216 + (alu269 + 4160)) = (((*(acc0 + 5))));
  *(data0_16777216 + (alu269 + 4176)) = (((*(acc0 + 13))));
  *(data0_16777216 + (alu269 + 4192)) = (((*(acc0 + 21))));
  *(data0_16777216 + (alu269 + 4208)) = (((*(acc0 + 29))));
  *(data0_16777216 + (alu269 + 8192)) = (((*(acc0 + 2))));
  *(data0_16777216 + (alu269 + 8208)) = (((*(acc0 + 10))));
  *(data0_16777216 + (alu269 + 8224)) = (((*(acc0 + 18))));
  *(data0_16777216 + (alu269 + 8240)) = (((*(acc0 + 26))));
  *(data0_16777216 + (alu269 + 8256)) = (((*(acc0 + 6))));
  *(data0_16777216 + (alu269 + 8272)) = (((*(acc0 + 14))));
  *(data0_16777216 + (alu269 + 8288)) = (((*(acc0 + 22))));
  *(data0_16777216 + (alu269 + 8304)) = (((*(acc0 + 30))));
  *(data0_16777216 + (alu269 + 12288)) = (((*(acc0 + 3))));
  *(data0_16777216 + (alu269 + 12304)) = (((*(acc0 + 11))));
  *(data0_16777216 + (alu269 + 12320)) = (((*(acc0 + 19))));
  *(data0_16777216 + (alu269 + 12336)) = (((*(acc0 + 27))));
  *(data0_16777216 + (alu269 + 12352)) = (((*(acc0 + 7))));
  *(data0_16777216 + (alu269 + 12368)) = (((*(acc0 + 15))));
  *(data0_16777216 + (alu269 + 12384)) = (((*(acc0 + 23))));
  *(data0_16777216 + (alu269 + 12400)) = (((*(acc0 + 31))));
  *(data0_16777216 + (alu269 + 65536)) = (((*(acc0 + 32))));
  *(data0_16777216 + (alu269 + 65552)) = (((*(acc0 + 40))));
  *(data0_16777216 + (alu269 + 65568)) = (((*(acc0 + 48))));
  *(data0_16777216 + (alu269 + 65584)) = (((*(acc0 + 56))));
  *(data0_16777216 + (alu269 + 65600)) = (((*(acc0 + 36))));
  *(data0_16777216 + (alu269 + 65616)) = (((*(acc0 + 44))));
  *(data0_16777216 + (alu269 + 65632)) = (((*(acc0 + 52))));
  *(data0_16777216 + (alu269 + 65648)) = (((*(acc0 + 60))));
  *(data0_16777216 + (alu269 + 69632)) = (((*(acc0 + 33))));
  *(data0_16777216 + (alu269 + 69648)) = (((*(acc0 + 41))));
  *(data0_16777216 + (alu269 + 69664)) = (((*(acc0 + 49))));
  *(data0_16777216 + (alu269 + 69680)) = (((*(acc0 + 57))));
  *(data0_16777216 + (alu269 + 69696)) = (((*(acc0 + 37))));
  *(data0_16777216 + (alu269 + 69712)) = (((*(acc0 + 45))));
  *(data0_16777216 + (alu269 + 69728)) = (((*(acc0 + 53))));
  *(data0_16777216 + (alu269 + 69744)) = (((*(acc0 + 61))));
  *(data0_16777216 + (alu269 + 73728)) = (((*(acc0 + 34))));
  *(data0_16777216 + (alu269 + 73744)) = (((*(acc0 + 42))));
  *(data0_16777216 + (alu269 + 73760)) = (((*(acc0 + 50))));
  *(data0_16777216 + (alu269 + 73776)) = (((*(acc0 + 58))));
  *(data0_16777216 + (alu269 + 73792)) = (((*(acc0 + 38))));
  *(data0_16777216 + (alu269 + 73808)) = (((*(acc0 + 46))));
  *(data0_16777216 + (alu269 + 73824)) = (((*(acc0 + 54))));
  *(data0_16777216 + (alu269 + 73840)) = (((*(acc0 + 62))));
  *(data0_16777216 + (alu269 + 77824)) = (((*(acc0 + 35))));
  *(data0_16777216 + (alu269 + 77840)) = (((*(acc0 + 43))));
  *(data0_16777216 + (alu269 + 77856)) = (((*(acc0 + 51))));
  *(data0_16777216 + (alu269 + 77872)) = (((*(acc0 + 59))));
  *(data0_16777216 + (alu269 + 77888)) = (((*(acc0 + 39))));
  *(data0_16777216 + (alu269 + 77904)) = (((*(acc0 + 47))));
  *(data0_16777216 + (alu269 + 77920)) = (((*(acc0 + 55))));
  *(data0_16777216 + (alu269 + 77936)) = (((*(acc0 + 63))));
  *(data0_16777216 + (alu269 + 131072)) = (((*(acc0 + 64))));
  *(data0_16777216 + (alu269 + 131088)) = (((*(acc0 + 72))));
  *(data0_16777216 + (alu269 + 131104)) = (((*(acc0 + 80))));
  *(data0_16777216 + (alu269 + 131120)) = (((*(acc0 + 88))));
  *(data0_16777216 + (alu269 + 131136)) = (((*(acc0 + 68))));
  *(data0_16777216 + (alu269 + 131152)) = (((*(acc0 + 76))));
  *(data0_16777216 + (alu269 + 131168)) = (((*(acc0 + 84))));
  *(data0_16777216 + (alu269 + 131184)) = (((*(acc0 + 92))));
  *(data0_16777216 + (alu269 + 135168)) = (((*(acc0 + 65))));
  *(data0_16777216 + (alu269 + 135184)) = (((*(acc0 + 73))));
  *(data0_16777216 + (alu269 + 135200)) = (((*(acc0 + 81))));
  *(data0_16777216 + (alu269 + 135216)) = (((*(acc0 + 89))));
  *(data0_16777216 + (alu269 + 135232)) = (((*(acc0 + 69))));
  *(data0_16777216 + (alu269 + 135248)) = (((*(acc0 + 77))));
  *(data0_16777216 + (alu269 + 135264)) = (((*(acc0 + 85))));
  *(data0_16777216 + (alu269 + 135280)) = (((*(acc0 + 93))));
  *(data0_16777216 + (alu269 + 139264)) = (((*(acc0 + 66))));
  *(data0_16777216 + (alu269 + 139280)) = (((*(acc0 + 74))));
  *(data0_16777216 + (alu269 + 139296)) = (((*(acc0 + 82))));
  *(data0_16777216 + (alu269 + 139312)) = (((*(acc0 + 90))));
  *(data0_16777216 + (alu269 + 139328)) = (((*(acc0 + 70))));
  *(data0_16777216 + (alu269 + 139344)) = (((*(acc0 + 78))));
  *(data0_16777216 + (alu269 + 139360)) = (((*(acc0 + 86))));
  *(data0_16777216 + (alu269 + 139376)) = (((*(acc0 + 94))));
  *(data0_16777216 + (alu269 + 143360)) = (((*(acc0 + 67))));
  *(data0_16777216 + (alu269 + 143376)) = (((*(acc0 + 75))));
  *(data0_16777216 + (alu269 + 143392)) = (((*(acc0 + 83))));
  *(data0_16777216 + (alu269 + 143408)) = (((*(acc0 + 91))));
  *(data0_16777216 + (alu269 + 143424)) = (((*(acc0 + 71))));
  *(data0_16777216 + (alu269 + 143440)) = (((*(acc0 + 79))));
  *(data0_16777216 + (alu269 + 143456)) = (((*(acc0 + 87))));
  *(data0_16777216 + (alu269 + 143472)) = (((*(acc0 + 95))));
  *(data0_16777216 + (alu269 + 196608)) = (((*(acc0 + 96))));
  *(data0_16777216 + (alu269 + 196624)) = (((*(acc0 + 104))));
  *(data0_16777216 + (alu269 + 196640)) = (((*(acc0 + 112))));
  *(data0_16777216 + (alu269 + 196656)) = (((*(acc0 + 120))));
  *(data0_16777216 + (alu269 + 196672)) = (((*(acc0 + 100))));
  *(data0_16777216 + (alu269 + 196688)) = (((*(acc0 + 108))));
  *(data0_16777216 + (alu269 + 196704)) = (((*(acc0 + 116))));
  *(data0_16777216 + (alu269 + 196720)) = (((*(acc0 + 124))));
  *(data0_16777216 + (alu269 + 200704)) = (((*(acc0 + 97))));
  *(data0_16777216 + (alu269 + 200720)) = (((*(acc0 + 105))));
  *(data0_16777216 + (alu269 + 200736)) = (((*(acc0 + 113))));
  *(data0_16777216 + (alu269 + 200752)) = (((*(acc0 + 121))));
  *(data0_16777216 + (alu269 + 200768)) = (((*(acc0 + 101))));
  *(data0_16777216 + (alu269 + 200784)) = (((*(acc0 + 109))));
  *(data0_16777216 + (alu269 + 200800)) = (((*(acc0 + 117))));
  *(data0_16777216 + (alu269 + 200816)) = (((*(acc0 + 125))));
  *(data0_16777216 + (alu269 + 204800)) = (((*(acc0 + 98))));
  *(data0_16777216 + (alu269 + 204816)) = (((*(acc0 + 106))));
  *(data0_16777216 + (alu269 + 204832)) = (((*(acc0 + 114))));
  *(data0_16777216 + (alu269 + 204848)) = (((*(acc0 + 122))));
  *(data0_16777216 + (alu269 + 204864)) = (((*(acc0 + 102))));
  *(data0_16777216 + (alu269 + 204880)) = (((*(acc0 + 110))));
  *(data0_16777216 + (alu269 + 204896)) = (((*(acc0 + 118))));
  *(data0_16777216 + (alu269 + 204912)) = (((*(acc0 + 126))));
  *(data0_16777216 + (alu269 + 208896)) = (((*(acc0 + 99))));
  *(data0_16777216 + (alu269 + 208912)) = (((*(acc0 + 107))));
  *(data0_16777216 + (alu269 + 208928)) = (((*(acc0 + 115))));
  *(data0_16777216 + (alu269 + 208944)) = (((*(acc0 + 123))));
  *(data0_16777216 + (alu269 + 208960)) = (((*(acc0 + 103))));
  *(data0_16777216 + (alu269 + 208976)) = (((*(acc0 + 111))));
  *(data0_16777216 + (alu269 + 208992)) = (((*(acc0 + 119))));
  *(data0_16777216 + (alu269 + 209008)) = (((*(acc0 + 127))));
}

int main() {
  std::mt19937 gen(0);
  std::uniform_real_distribution<float> dist(-1, 1);

  assert(N % inst_size == 0);

  std::cout << "Generating A matrix..." << std::endl;
  std::vector<half> A_h(A_size);
  for (int i = 0; i < A_h.size(); ++i) {
    A_h[i] = static_cast<half>(dist(gen));
  }
  std::cout << "Generating B matrix..." << std::endl;
  std::vector<half> B_h(B_size);
  for (int i = 0; i < B_h.size(); ++i) {
    B_h[i] = static_cast<half>(dist(gen));
  }

  std::vector<float> D_h(D_size);

  // std::cout << "Calculating on host..." << std::endl;
  std::vector<float> Dref_h(D_size);
  // gemm_host(A_h, B_h, Dref_h, N, N, N, N, N, N);

  std::cout << "Allocating GPU buffers..." << std::endl;
  half *A_d, *B_d;
  float *D_d;
  HIP_CHECK(hipMalloc(&A_d, A_size * sizeof(half)));
  HIP_CHECK(hipMalloc(&B_d, B_size * sizeof(half)));
  HIP_CHECK(hipMalloc(&D_d, D_size * sizeof(float)));
  HIP_CHECK(
      hipMemcpy(A_d, A_h.data(), A_size * sizeof(half), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(B_d, B_h.data(), B_size * sizeof(half), hipMemcpyHostToDevice));
  // HIP_CHECK(hipMemcpy(D_d, D_h.data(), D_size * sizeof(half),
  //                     hipMemcpyHostToDevice));

  std::cout << "Launching GPU kernel..." << std::endl;
  r_64_4_64_4_2_2_2_4_4_2_256<<<dim3(4, 64), dim3(64, 4, 2)>>>(D_d, A_d, B_d);
  r_64_4_64_4_2_2_2_4_4_2_256<<<dim3(4, 64), dim3(64, 4, 2)>>>(D_d, A_d, B_d);
  r_64_4_64_4_2_2_2_4_4_2_256<<<dim3(4, 64), dim3(64, 4, 2)>>>(D_d, A_d, B_d);
  r_64_4_64_4_2_2_2_4_4_2_256<<<dim3(4, 64), dim3(64, 4, 2)>>>(D_d, A_d, B_d);
  r_64_4_64_4_2_2_2_4_4_2_256<<<dim3(4, 64), dim3(64, 4, 2)>>>(D_d, A_d, B_d);
  r_64_4_64_4_2_2_2_4_4_2_256<<<dim3(4, 64), dim3(64, 4, 2)>>>(D_d, A_d, B_d);
  r_64_4_64_4_2_2_2_4_4_2_256<<<dim3(4, 64), dim3(64, 4, 2)>>>(D_d, A_d, B_d);
  HIP_CHECK(hipGetLastError());

  std::cout << "Copying result from GPU..." << std::endl;
  HIP_CHECK(hipMemcpy(D_h.data(), D_d, D_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  // std::cout << "Sum of squared differences of host/device result matrices: "
  //           << compute_l2_error(Dref_h, D_h, N, N, N, N) << std::endl;

  HIP_CHECK(hipFree(D_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(A_d));
  return 0;
}
