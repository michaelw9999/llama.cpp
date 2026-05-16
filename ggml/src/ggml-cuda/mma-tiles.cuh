#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "ggml.h"
#include "ggml-common.h"
#ifndef GGML_HD
#define GGML_HD __host__ __device__
#endif // GGML_HD

static_assert(sizeof(tile_mxfp6_frag) == MXFP6_TILE_BYTES / MXFP6_TILE_FRAGS, "unexpected MXFP6 fragment size");
static_assert(sizeof(tile_mxfp6) == MXFP6_TILE_BYTES, "unexpected MXFP6 tile size");
static_assert(sizeof(tensor_mxfp6) == 32, "unexpected MXFP6 tensor header size");
static_assert(alignof(tile_mxfp6_frag) == 16, "MXFP6 fragment must be 16B aligned");
static_assert(alignof(tile_mxfp6) == 16, "MXFP6 tile must be 16B aligned");
static_assert(alignof(tensor_mxfp6) == 16, "MXFP6 tensor header must be 16B aligned");

static inline GGML_HD int64_t ggml_cuda_mxfp6_e2m3_frags_per_row(int64_t ne0) {
    return (ne0 + QK_MXFP6_E2M3 - 1) / QK_MXFP6_E2M3;
}

static inline GGML_HD uint32_t ggml_cuda_mxfp6_e2m3_codes_to_lanes(uint32_t packed) {
    packed &= 0x00FFFFFFu;
    return ((packed & 0x00003Fu) << 0) |
           ((packed & 0x000FC0u) << 2) |
           ((packed & 0x03F000u) << 4) |
           ((packed & 0xFC0000u) << 6);
}

static inline GGML_HD uint32_t ggml_cuda_mxfp6_e2m3_frag_codes_to_lanes(
        const tile_mxfp6_frag & frag, int row_in_tile, int pack_idx) {
    const int lane = ((row_in_tile & 7) * 4) + (pack_idx & 3);
    const uint32_t w0 = frag.lane[lane][0];
    const uint32_t w1 = frag.lane[lane][1];
    const uint32_t w2 = frag.lane[lane][2];
    const uint32_t packed = (row_in_tile & 8) ?
        ((pack_idx & 4) ? (w2 >> 8) : ((w0 >> 24) | (w1 << 8))) :
        ((pack_idx & 4) ? ((w1 >> 16) | (w2 << 16)) : w0);
    return ggml_cuda_mxfp6_e2m3_codes_to_lanes(packed);
}

static inline size_t ggml_cuda_mxfp6_e2m3_plane_size(int64_t ne0, int64_t nrows) {
    return (size_t) ((nrows + MXFP6_TILE_ROWS - 1) / MXFP6_TILE_ROWS) *
           (size_t) ggml_cuda_mxfp6_e2m3_frags_per_row(ne0) * sizeof(tile_mxfp6_frag);
}

static inline bool ggml_cuda_mxfp6_e2m3_get_scalar_f32(const ggml_tensor * tensor, float * value) {
    if (tensor == nullptr || !ggml_is_scalar(tensor) || tensor->type != GGML_TYPE_F32 || tensor->data == nullptr) {
        return false;
    }
    if (tensor->buffer == nullptr || ggml_backend_buffer_is_host(tensor->buffer)) {
        memcpy(value, tensor->data, sizeof(*value));
        return true;
    }
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    return cudaMemcpy(value, tensor->data, sizeof(*value), cudaMemcpyDeviceToHost) == cudaSuccess;
#else
    return false;
#endif
}

static inline const float * ggml_cuda_mxfp6_e2m3_scale_ptr(const ggml_tensor * tensor) {
    return tensor != nullptr && !ggml_is_scalar(tensor) && tensor->type == GGML_TYPE_F32 &&
        tensor->data != nullptr && tensor->buffer != nullptr &&
        !ggml_backend_buffer_is_host(tensor->buffer) ? (const float *) tensor->data : nullptr;
}

static inline void ggml_cuda_mxfp6_e2m3_patch_tensor_header(const ggml_tensor * tensor, tensor_mxfp6 * dst) {
    const ggml_tensor * weight_scale_t = tensor != nullptr ? tensor->src[0] : nullptr;
    const ggml_tensor * input_scale_t  = tensor != nullptr ? tensor->src[1] : nullptr;
    float weight_scale = dst->weight_scale;
    float input_scale  = dst->input_scale;

    if (ggml_cuda_mxfp6_e2m3_get_scalar_f32(weight_scale_t, &weight_scale) && weight_scale > 0.0f) {
        dst->weight_scale = weight_scale;
    }
    if (ggml_cuda_mxfp6_e2m3_get_scalar_f32(input_scale_t, &input_scale) && input_scale > 0.0f) {
        dst->input_scale = input_scale;
    }
    if (!(dst->weight_scale > 0.0f)) {
        dst->weight_scale = 1.0f;
    }
    if (!(dst->input_scale > 0.0f)) {
        dst->input_scale = 1.0f;
    }
    dst->weight_scales = ggml_cuda_mxfp6_e2m3_scale_ptr(weight_scale_t);
    dst->input_scales  = ggml_cuda_mxfp6_e2m3_scale_ptr(input_scale_t);
}
