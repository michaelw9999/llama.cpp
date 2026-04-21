#include "quantize.cuh"
#include <cstdint>

__launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1)
static __global__ void quantize_q8_1(
        const float * __restrict__ x, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const uint32_t ne1, const uint3 ne2) {
    const int64_t i0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i3 = fastdiv(blockIdx.z, ne2);
    const int64_t i2 = blockIdx.z - i3*ne2.z;
    const int64_t i1 = blockIdx.y;

    const int64_t & i00 = i0;
    const int64_t & i01 = i1;
    const int64_t & i02 = i2;
    const int64_t & i03 = i3;

    const int64_t i_cont = ((i3*ne2.z + i2) * ne1 + i1) * ne0 + i0;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int64_t ib  = i_cont / QK8_1; // block index
    const int64_t iqs = i_cont % QK8_1; // quant index

    const float xi = i0 < ne00 ? x[i03*s03 + i02*s02 + i01*s01 + i00] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    y[ib].ds = make_half2(d, sum);
}

__device__ __forceinline__ uint8_t compute_e8m0_scale(float amax) {
    if (!(amax > 0.0f)) {
        return 0;
    }

    // FP4 E2M1: max exponent (unbiased) is 2.
    constexpr int FP4_E2M1_EMAX = 2;

    const float e = log2f(amax);

    // "even" -> round-to-nearest integer, ties-to-even
    const int e_int = __float2int_rn(e);

    const int shared_exp = e_int - FP4_E2M1_EMAX;

    int biased = shared_exp + 127;

    biased = max(biased, 0);
    biased = min(biased, 254);

    return static_cast<uint8_t>(biased);
}

template <bool has_ids, bool channel_major>
static __global__ void quantize_mmq_nvfp4(const float * __restrict__ x,
                                          const int32_t * __restrict__ ids,
                                          void * __restrict__ vy,
                                          const int64_t ne00,
                                          const int64_t s01,
                                          const int64_t s02,
                                          const int64_t s03,
                                          const int64_t ne0,
                                          const int64_t ne1,
                                          const int64_t ne2) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    const int64_t i0_base = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * 8;
    if (i0_base >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;
    const int64_t i01 = has_ids ? ids[i1] : i1;

    const int64_t warp_base = ((int64_t) blockDim.x * blockIdx.y + warp_id * 32) * 8;
    const int64_t k_block = warp_base / QK_K;
    const int64_t blocks_per_col = (ne0 + QK_K - 1) / QK_K;
    if (k_block >= blocks_per_col) {
        return;
    }

    const int64_t batch_offset = (int64_t) blockIdx.z * (blocks_per_col * ne1);
    // CRITICAL: DO NOT change this without checking both callers.
    // MMQ still consumes [batch][k_block][channel], while MMVQ packed NVFP4 consumes [batch][channel][k_block].
    const int64_t ib = channel_major
        ? batch_offset + blockIdx.x * blocks_per_col + k_block
        : batch_offset + k_block * ne1 + blockIdx.x;
    block_nvfp4_mmq * y_nv = (block_nvfp4_mmq *) vy;
    block_nvfp4_mmq * yb = y_nv + ib;

    float vals_raw[8];
    float amax_raw = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int64_t i00 = i0_base + k;
        if (i00 < ne00) {
            const int64_t idx = i3 * s03 + i2 * s02 + i01 * s01 + i00;
            float v = x[idx];
            vals_raw[k] = v;
            amax_raw = fmaxf(amax_raw, fabsf(v));
        } else {
            vals_raw[k] = 0.0f;
        }
    }

    const unsigned lane_pair = lane_id & ~1u;
    const float sub_max = fmaxf(amax_raw, __shfl_xor_sync(0xFFFFFFFFu, amax_raw, 1));
    uint32_t fp8_code_u32 = 0;
    float subblock_scale = 0.0f;
    if ((lane_id & 1) == 0) {
        uint8_t fp8_code = ggml_cuda_fp32_to_ue4m3(sub_max * (1.0f / 6.0f));
        subblock_scale = ggml_cuda_ue4m3_to_fp32(fp8_code);

        if (!(subblock_scale > 0.0f) || !isfinite(subblock_scale)) {
            subblock_scale = 0.0f;
            fp8_code = 0;
        }
        fp8_code_u32 = fp8_code;
    }
    fp8_code_u32 = __shfl_sync(0xFFFFFFFFu, fp8_code_u32, lane_pair);
    subblock_scale = __shfl_sync(0xFFFFFFFFu, subblock_scale, lane_pair);
    uint8_t fp8_code = (uint8_t) fp8_code_u32;

    uchar4 packed;

    const float inv_scale = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;
    __nv_fp4x4_e2m1 packed0(make_float4(vals_raw[0] * inv_scale,
                                        vals_raw[1] * inv_scale,
                                        vals_raw[2] * inv_scale,
                                        vals_raw[3] * inv_scale));
    __nv_fp4x4_e2m1 packed1(make_float4(vals_raw[4] * inv_scale,
                                        vals_raw[5] * inv_scale,
                                        vals_raw[6] * inv_scale,
                                        vals_raw[7] * inv_scale));
    const char2 bytes0 = *(const char2 *) &packed0;
    const char2 bytes1 = *(const char2 *) &packed1;
    packed = make_uchar4((uint8_t) bytes0.x, (uint8_t) bytes0.y, (uint8_t) bytes1.x, (uint8_t) bytes1.y);

    ((uchar4 *) yb->qs_u32)[lane_id] = packed;

    if ((lane_id & 1) == 0) {
        reinterpret_cast<uint8_t *>(yb->sc4_u32)[lane_id >> 1] = fp8_code;
    }
}

static __global__ void quantize_row_nvfp4(const float * __restrict__ x,
                                          void * __restrict__ vy,
                                          const int64_t ne00,
                                          const int64_t s01,
                                          const int64_t s02,
                                          const int64_t s03,
                                          const int64_t ne0,
                                          const int64_t ne1,
                                          const int64_t ne2) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    const int64_t i0_base = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * 8;
    if (i0_base >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t warp_base = ((int64_t) blockDim.x * blockIdx.y + warp_id * 32) * 8;
    const int64_t k_block = warp_base / QK_K;
    const int64_t blocks_per_col = (ne0 + QK_K - 1) / QK_K;
    if (k_block >= blocks_per_col) {
        return;
    }

    const int64_t batch_offset = ((int64_t) i3 * ne2 + i2) * (ne1 * blocks_per_col);
    const int64_t ib = batch_offset + i1 * blocks_per_col + k_block;
    block_nvfp4_mmq * y_nv = (block_nvfp4_mmq *) vy;
    block_nvfp4_mmq * yb = y_nv + ib;

    float vals_raw[8];
    float amax_raw = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int64_t i00 = i0_base + k;
        if (i00 < ne00) {
            const int64_t idx = i3 * s03 + i2 * s02 + i1 * s01 + i00;
            float v = x[idx];
            vals_raw[k] = v;
            amax_raw = fmaxf(amax_raw, fabsf(v));
        } else {
            vals_raw[k] = 0.0f;
        }
    }

    const unsigned lane_pair = lane_id & ~1u;
    const float sub_max = fmaxf(amax_raw, __shfl_xor_sync(0xFFFFFFFFu, amax_raw, 1));
    uint32_t fp8_code_u32 = 0;
    float subblock_scale = 0.0f;
    if ((lane_id & 1) == 0) {
        uint8_t fp8_code = ggml_cuda_fp32_to_ue4m3(sub_max * (1.0f / 6.0f));
        subblock_scale = ggml_cuda_ue4m3_to_fp32(fp8_code);

        if (!(subblock_scale > 0.0f) || !isfinite(subblock_scale)) {
            subblock_scale = 0.0f;
            fp8_code = 0;
        }
        fp8_code_u32 = fp8_code;
    }
    fp8_code_u32 = __shfl_sync(0xFFFFFFFFu, fp8_code_u32, lane_pair);
    subblock_scale = __shfl_sync(0xFFFFFFFFu, subblock_scale, lane_pair);
    const uint8_t fp8_code = (uint8_t) fp8_code_u32;

    uchar4 packed;
    const float inv_scale = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;
    __nv_fp4x4_e2m1 packed0(make_float4(
        vals_raw[0] * inv_scale,
        vals_raw[1] * inv_scale,
        vals_raw[2] * inv_scale,
        vals_raw[3] * inv_scale));
    __nv_fp4x4_e2m1 packed1(make_float4(
        vals_raw[4] * inv_scale,
        vals_raw[5] * inv_scale,
        vals_raw[6] * inv_scale,
        vals_raw[7] * inv_scale));
    const char2 bytes0 = *(const char2 *) &packed0;
    const char2 bytes1 = *(const char2 *) &packed1;
    packed = make_uchar4((uint8_t) bytes0.x, (uint8_t) bytes0.y, (uint8_t) bytes1.x, (uint8_t) bytes1.y);


    ((uchar4 *) yb->qs_u32)[lane_id] = packed;

    if ((lane_id & 1) == 0) {
        reinterpret_cast<uint8_t *>(yb->sc4_u32)[lane_id >> 1] = fp8_code;
    }
}

template <bool has_ids>
static __global__ void quantize_mmq_nvfp4_scaled(const float * __restrict__ x,
                                                 const int32_t * __restrict__ ids,
                                                 const int32_t * __restrict__ ids_expert,
                                                 void * __restrict__ vy,
                                                 const float * __restrict__ input_scale,
                                                 const int64_t input_scale_ne,
                                                 const int64_t ne00,
                                                 const int64_t s01,
                                                 const int64_t s02,
                                                 const int64_t s03,
                                                 const int64_t ne0,
                                                 const int64_t ne1,
                                                 const int64_t ne2) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    const int64_t i0_base = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * 8;
    if (i0_base >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;
    const int64_t i01 = has_ids ? ids[i1] : i1;

    const int64_t warp_base = ((int64_t) blockDim.x * blockIdx.y + warp_id * 32) * 8;
    const int64_t k_block = warp_base / QK_K;
    const int64_t blocks_per_col = (ne0 + QK_K - 1) / QK_K;
    if (k_block >= blocks_per_col) {
        return;
    }

    const int64_t batch_offset = (int64_t) blockIdx.z * (blocks_per_col * ne1);
    const int64_t ib = batch_offset + k_block * ne1 + blockIdx.x;
    block_nvfp4_mmq * y_nv = (block_nvfp4_mmq *) vy;
    block_nvfp4_mmq * yb = y_nv + ib;

    const int scale_idx = input_scale_ne > 1 ? (has_ids ? ids_expert[i1] : i01) : 0;
    const float input_s = input_scale ? input_scale[scale_idx] : 1.0f;
    GGML_ASSERT(input_s != 0.0f);
    const float inv_input_s = 1.0f / input_s;

    float vals_raw[8];
    float amax_raw = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int64_t i00 = i0_base + k;
        if (i00 < ne00) {
            const int64_t idx = i3 * s03 + i2 * s02 + i01 * s01 + i00;
            float v = x[idx] * inv_input_s;
            vals_raw[k] = v;
            amax_raw = fmaxf(amax_raw, fabsf(v));
        } else {
            vals_raw[k] = 0.0f;
        }
    }

    const unsigned lane_pair = lane_id & ~1u;
    const float sub_max = fmaxf(amax_raw, __shfl_xor_sync(0xFFFFFFFFu, amax_raw, 1));
    uint32_t fp8_code_u32 = 0;
    float subblock_scale = 0.0f;
    if ((lane_id & 1) == 0) {
        uint8_t fp8_code = ggml_cuda_fp32_to_ue4m3(sub_max * (1.0f / 6.0f));
        subblock_scale = ggml_cuda_ue4m3_to_fp32(fp8_code);

        if (!(subblock_scale > 0.0f) || !isfinite(subblock_scale)) {
            subblock_scale = 0.0f;
            fp8_code = 0;
        }
        fp8_code_u32 = fp8_code;
    }
    fp8_code_u32 = __shfl_sync(0xFFFFFFFFu, fp8_code_u32, lane_pair);
    subblock_scale = __shfl_sync(0xFFFFFFFFu, subblock_scale, lane_pair);
    uint8_t fp8_code = (uint8_t) fp8_code_u32;

    const float inv_scale = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;
    __nv_fp4x4_e2m1 packed0(make_float4(vals_raw[0] * inv_scale,
                                        vals_raw[1] * inv_scale,
                                        vals_raw[2] * inv_scale,
                                        vals_raw[3] * inv_scale));
    __nv_fp4x4_e2m1 packed1(make_float4(vals_raw[4] * inv_scale,
                                        vals_raw[5] * inv_scale,
                                        vals_raw[6] * inv_scale,
                                        vals_raw[7] * inv_scale));
    const char2 bytes0 = *(const char2 *) &packed0;
    const char2 bytes1 = *(const char2 *) &packed1;
    const uchar4 packed = make_uchar4((uint8_t) bytes0.x, (uint8_t) bytes0.y, (uint8_t) bytes1.x, (uint8_t) bytes1.y);

    ((uchar4 *) yb->qs_u32)[lane_id] = packed;

    if ((lane_id & 1) == 0) {
        reinterpret_cast<uint8_t *>(yb->sc4_u32)[lane_id >> 1] = fp8_code;
    }
}

// quantize values in the format mxfp4 is stored which is interleaved nibbles
// i.e. a block a0-a31 is represented as a0a16,a1a17 ...a15a31
static __global__ void quantize_mmq_mxfp4(const float * __restrict__ x,
                                          const int32_t * __restrict__ ids,
                                          void * __restrict__ vy,
                                          const int64_t ne00,
                                          const int64_t s01,
                                          const int64_t s02,
                                          const int64_t s03,
                                          const int64_t ne0,
                                          const int     ne1,
                                          const int     ne2) {
    constexpr int vals_per_scale = 32;
    constexpr int vals_per_warp  = 2 * vals_per_scale;  // Each warp processes 2 blocks of 32 = 64 values

    const int warp_id = threadIdx.y;
    const int lane_id_32 = threadIdx.x;

    const int nwarps = blockDim.y;

    const int64_t warp_start_offset = (blockIdx.y * nwarps + warp_id) * vals_per_warp;

    if (warp_start_offset >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    block_fp4_mmq * y = (block_fp4_mmq *) vy;

    const int64_t block_fp4_mmq_size = 8 * QK_MXFP4;  // 256 values
    const int64_t ib0                = blockIdx.z * ((int64_t) ne1 * (ne0 / block_fp4_mmq_size));
    const int64_t ib = ib0 + (warp_start_offset / block_fp4_mmq_size) * ne1 + blockIdx.x;
    const int64_t quad_idx_in_block  = (warp_start_offset % block_fp4_mmq_size) / vals_per_warp;

    const int group_id = lane_id_32 / 4;
    const int lane_in_group = lane_id_32 % 4;
    const int base = group_id * 2;
    char2 * yqs2 = (char2 *) y[ib].qs;

    const int64_t base_pos = i03 * s03 + i02 * s02 + i01 * s01;

    uint8_t scales[2];

#pragma unroll
    for (int b = 0; b < 2; ++b) {
        const int64_t i0 = warp_start_offset + b * vals_per_scale + lane_id_32;
        const float xi = (i0 < ne00) ? x[base_pos + i0] : 0.0f;

        float amax = fabsf(xi);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, WARP_SIZE));
        }

        const uint8_t e = compute_e8m0_scale(amax);
        scales[b] = e;
        const float inv_s = (amax == 0.0f) ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(e));

#if CUDART_VERSION >= 12080
        const float scaled_val = xi * inv_s;

        const float val0 = __shfl_sync(0xFFFFFFFF, scaled_val, base, WARP_SIZE);
        const float val1 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 16, WARP_SIZE);
        const float val2 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 1, WARP_SIZE);
        const float val3 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 17, WARP_SIZE);

        if (lane_in_group == 0) {
            __nv_fp4x4_e2m1 fp4_packed(make_float4(val0, val1, val2, val3));

            yqs2[quad_idx_in_block * 16 + b * 8 + group_id] = *(char2 *) &fp4_packed;
        }
#else
        // Fallback: manual FP4 conversion using LUT
        const uint8_t q_val = ggml_cuda_float_to_fp4_e2m1(xi, inv_s);

        const uint8_t q_lo_0 = __shfl_sync(0xFFFFFFFF, q_val, base,      WARP_SIZE);
        const uint8_t q_lo_1 = __shfl_sync(0xFFFFFFFF, q_val, base + 1,  WARP_SIZE);
        const uint8_t q_hi_0 = __shfl_sync(0xFFFFFFFF, q_val, base + 16, WARP_SIZE);
        const uint8_t q_hi_1 = __shfl_sync(0xFFFFFFFF, q_val, base + 17, WARP_SIZE);

        if (lane_in_group == 0) {
            char2 q;
            q.x = (q_hi_0 << 4) | q_lo_0;
            q.y = (q_hi_1 << 4) | q_lo_1;
            yqs2[quad_idx_in_block * 16 + b * 8 + group_id] = q;
        }
#endif // CUDART_VERSION >= 12080
    }

    if (lane_id_32 == 0) {
        // Store 2 scales packed into 1 uint32
        y[ib].d4[quad_idx_in_block] = (scales[1] << 8) | scales[0];
    }
}

template <mmq_q8_1_ds_layout ds_layout>
static __global__ void quantize_mmq_q8_1(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t i00 = i0;
    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    const float4 * x4 = (const float4 *) x;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1); // first block of channel
    const int64_t ib  = ib0 + (i0 / (4*QK8_1))*ne1 + blockIdx.x;                    // block index in channel
    const int64_t iqs = i0 % (4*QK8_1);                                             // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them:
    const float4 xi = i0 < ne00 ? x4[(i03*s03 + i02*s02 + i01*s01 + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;

        // Calculate sums across vals_per_sum/4 threads.
#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);

    // Write back 4 int8 values as a single 32 bit value for better memory bandwidth:
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 != 0 || iqs >= 96) {
            return;
        }

        y[ib].d2s6[2 + iqs/16] = sum;

        if (iqs % 64 != 0) {
            return;
        }

        const float d = 1.0f / d_inv;

        y[ib].d2s6[iqs/64] = d;

        return;
    }

    if (iqs % 32 != 0) {
        return;
    }

    const float d = 1.0f / d_inv;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        y[ib].ds4[iqs/32] = make_half2(d, sum);
    } else {
        y[ib].d4[iqs/32]  = d;
    }
}

void quantize_row_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(!ids);
    GGML_ASSERT(ne0 % QK8_1 == 0);

    const uint3 ne2_fastdiv = init_fastdiv_values(ne2);

    const int64_t block_num_x = (ne0 + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, ne1, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(x, vy, ne00, s01, s02, s03, ne0, ne1, ne2_fastdiv);
    GGML_UNUSED(type_src0);
}

void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % (4*QK8_1) == 0);

    // ne1 tends to assume the highest values, therefore use it as the "x" dimension of the CUDA grid:
    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void quantize_mmq_nvfp4_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(type_src0 == GGML_TYPE_NVFP4);
    GGML_ASSERT(ne00 % 8 == 0);
    GGML_ASSERT(ne0 > 0);

    constexpr int nvfp4_block_size = 128;
    const int64_t block_num_y = (ne0 + 8 * nvfp4_block_size - 1) / (8 * nvfp4_block_size);
    const dim3 num_blocks(ne1, block_num_y, ne2 * ne3);
    const dim3 block_size(nvfp4_block_size, 1, 1);
    if (ids) {
        quantize_mmq_nvfp4<true, false><<<num_blocks, block_size, 0, stream>>>(
            x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
    } else {
        quantize_mmq_nvfp4<false, false><<<num_blocks, block_size, 0, stream>>>(
            x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
    }
}

void quantize_mmvq_nvfp4_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(type_src0 == GGML_TYPE_NVFP4);
    GGML_ASSERT(ne00 % 8 == 0);
    GGML_ASSERT(ne0 > 0);

    constexpr int nvfp4_block_size = 32;
    const int64_t block_num_y = (ne0 + 8 * nvfp4_block_size - 1) / (8 * nvfp4_block_size);
    const dim3 num_blocks(ne1, block_num_y, ne2 * ne3);
    const dim3 block_size(nvfp4_block_size, 1, 1);
    if (ids) {
        quantize_mmq_nvfp4<true, true><<<num_blocks, block_size, 0, stream>>>(
            x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
    } else {
        quantize_mmq_nvfp4<false, true><<<num_blocks, block_size, 0, stream>>>(
            x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
    }
}

void quantize_row_nvfp4_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(type_src0 == GGML_TYPE_NVFP4);
    GGML_ASSERT(ids == nullptr);
    GGML_ASSERT(ne00 % 8 == 0);
    GGML_ASSERT(ne0 > 0);

    constexpr int nvfp4_block_size = 32;
    const int64_t block_num_y = (ne0 + 8 * nvfp4_block_size - 1) / (8 * nvfp4_block_size);
    const dim3 num_blocks(ne1, block_num_y, ne2 * ne3);
    const dim3 block_size(nvfp4_block_size, 1, 1);
    quantize_row_nvfp4<<<num_blocks, block_size, 0, stream>>>(
        x, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
}

void quantize_mmq_mxfp4_cuda(const float *                    x,
                             const int32_t *                  ids,
                             void *                           vy,
                             [[maybe_unused]] const ggml_type type_src0,
                             const int64_t                    ne00,
                             const int64_t                    s01,
                             const int64_t                    s02,
                             const int64_t                    s03,
                             const int64_t                    ne0,
                             const int64_t                    ne1,
                             const int64_t                    ne2,
                             const int64_t                    ne3,
                             cudaStream_t                     stream) {
    GGML_ASSERT(ne0 % (2 * QK_MXFP4) == 0);

    constexpr int nwarps = 8;
    constexpr int vals_per_warp  = 2 * QK_MXFP4;
    constexpr int vals_per_block = nwarps * vals_per_warp;

    const int64_t block_num_y = (ne0 + vals_per_block - 1) / vals_per_block;
    const dim3    num_blocks(ne1, block_num_y, ne2 * ne3);
    const dim3    block_size(WARP_SIZE, nwarps, 1);

    quantize_mmq_mxfp4<<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
}

template <bool has_ids>
void quantize_mmq_fp4_cuda(
        const float * x, const int32_t * ids, const int32_t * ids_expert, void * vy,
        const ggml_type type_src0, const ggml_tensor * input_scale,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    if (type_src0 == GGML_TYPE_MXFP4) {
        if constexpr (has_ids) {
            quantize_mmq_mxfp4_cuda(x, ids, vy, type_src0, ne00, s01, s02, s03, ne0, ne1, ne2, ne3, stream);
        } else {
            quantize_mmq_mxfp4_cuda(x, nullptr, vy, type_src0, ne00, s01, s02, s03, ne0, ne1, ne2, ne3, stream);
        }
        return;
    }

    GGML_ASSERT(type_src0 == GGML_TYPE_NVFP4);
    GGML_ASSERT(ne00 % 8 == 0);
    GGML_ASSERT(ne0 > 0);

    const float * input_scale_data = input_scale ? (const float *) input_scale->data : nullptr;
    const int64_t input_scale_ne = input_scale ? ggml_nelements(input_scale) : 0;

    constexpr int nvfp4_block_size = 128;
    const int64_t block_num_y = (ne0 + 8 * nvfp4_block_size - 1) / (8 * nvfp4_block_size);
    const dim3 num_blocks(ne1, block_num_y, ne2 * ne3);
    const dim3 block_size(nvfp4_block_size, 1, 1);
    quantize_mmq_nvfp4_scaled<has_ids><<<num_blocks, block_size, 0, stream>>>(
        x, ids, ids_expert, vy, input_scale_data, input_scale_ne, ne00, s01, s02, s03, ne0, ne1, ne2);
}

template void quantize_mmq_fp4_cuda<false>(
        const float * x, const int32_t * ids, const int32_t * ids_expert, void * vy,
        ggml_type type_src0, const ggml_tensor * input_scale,
        int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);
template void quantize_mmq_fp4_cuda<true>(
        const float * x, const int32_t * ids, const int32_t * ids_expert, void * vy,
        ggml_type type_src0, const ggml_tensor * input_scale,
        int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);
