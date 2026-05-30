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

struct __align__(16) block_nvfp4_blackwell_frag {
    uint32_t regs[32][4];
    uint32_t scales_u32[32];
};

struct __align__(16) block_nvfp4_blackwell {
    block_nvfp4_blackwell_frag tiles[4];
};

struct __align__(16) block_nvfp4_blackwell_tensor {
    float         weight_scale;
    float         input_scale;
    const float * weight_scales;
    const float * input_scales;
    block_nvfp4_blackwell tiles[];
};

static_assert(sizeof(block_nvfp4_blackwell_frag) == 640, "unexpected NVFP4 Blackwell fragment size");
static_assert(sizeof(block_nvfp4_blackwell) == 4*sizeof(block_nvfp4_blackwell_frag), "unexpected NVFP4 Blackwell tile size");
static_assert(sizeof(block_nvfp4_blackwell_tensor) == 32, "unexpected NVFP4 Blackwell tensor header size");
static_assert(alignof(block_nvfp4_blackwell_frag) == 16, "NVFP4 Blackwell fragment must be 16B aligned");
static_assert(alignof(block_nvfp4_blackwell) == 16, "NVFP4 Blackwell tile must be 16B aligned");
static_assert(alignof(block_nvfp4_blackwell_tensor) == 16, "NVFP4 Blackwell tensor header must be 16B aligned");

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

static inline GGML_HD uint32_t ggml_cuda_nvfp4_tile_q_word(
        const block_nvfp4_blackwell & tile, int row_in_tile, int frag_idx, int pack_idx) {
    const int lane = ((row_in_tile & 7) * 4) + (pack_idx & 3);
    const int reg  = (row_in_tile >> 3) + ((pack_idx >> 2) << 1);
    return tile.tiles[frag_idx].regs[lane][reg];
}

static inline GGML_HD uint32_t ggml_cuda_nvfp4_tile_scale_word(
        const block_nvfp4_blackwell & tile, int row_in_tile, int frag_idx) {
    const int lane = ((row_in_tile & 7) * 4) + (row_in_tile >> 3);
    return tile.tiles[frag_idx].scales_u32[lane];
}

static inline uint32_t ggml_cuda_bw_pack8(const uint8_t * p, int shift) {
    return
        (((uint32_t) ((p[0] >> shift) & 0x0F)) <<  0) |
        (((uint32_t) ((p[1] >> shift) & 0x0F)) <<  4) |
        (((uint32_t) ((p[2] >> shift) & 0x0F)) <<  8) |
        (((uint32_t) ((p[3] >> shift) & 0x0F)) << 12) |
        (((uint32_t) ((p[4] >> shift) & 0x0F)) << 16) |
        (((uint32_t) ((p[5] >> shift) & 0x0F)) << 20) |
        (((uint32_t) ((p[6] >> shift) & 0x0F)) << 24) |
        (((uint32_t) ((p[7] >> shift) & 0x0F)) << 28);
}

static inline void ggml_cuda_bw_unpack8(uint32_t q_lo, uint32_t q_hi, uint8_t * p) {
    for (int i = 0; i < 8; ++i) {
        p[i] = (uint8_t) (((q_lo >> (4*i)) & 0x0Fu) | (((q_hi >> (4*i)) & 0x0Fu) << 4));
    }
}

static inline GGML_HD int64_t ggml_cuda_nvfp4_blocks_per_row(int64_t ne0) {
    return (ne0 + QK_K - 1) / QK_K;
}

static inline size_t ggml_cuda_nvfp4_plane_size(int64_t ne0, int64_t nrows) {
    return (size_t) ((nrows + 15) / 16) *
           (size_t) ggml_cuda_nvfp4_blocks_per_row(ne0) * sizeof(block_nvfp4_blackwell);
}

static inline size_t ggml_cuda_nvfp4_rows_size(int64_t ne0, int64_t ne1, int64_t nplanes) {
    return (size_t) nplanes * (size_t) ne1 * ggml_row_size(GGML_TYPE_NVFP4, ne0);
}

static inline size_t ggml_cuda_nvfp4_tensor_packed_size(int64_t ne0, int64_t ne1, int64_t nplanes) {
    return sizeof(block_nvfp4_blackwell_tensor) + (size_t) nplanes * ggml_cuda_nvfp4_plane_size(ne0, ne1);
}

static inline size_t ggml_cuda_nvfp4_tensor_size(int64_t ne0, int64_t ne1, int64_t nplanes) {
    return ggml_cuda_nvfp4_tensor_packed_size(ne0, ne1, nplanes);
}

static inline size_t ggml_cuda_nvfp4_tensor_alloc_size(const ggml_tensor * tensor) {
    return ggml_cuda_nvfp4_tensor_size(tensor->ne[0], tensor->ne[1], tensor->ne[2] * tensor->ne[3]);
}

static inline bool ggml_cuda_nvfp4_get_scalar_f32(const ggml_tensor * tensor, float * value) {
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

static inline const float * ggml_cuda_nvfp4_scale_ptr(const ggml_tensor * tensor) {
    return tensor != nullptr && !ggml_is_scalar(tensor) && tensor->type == GGML_TYPE_F32 &&
        tensor->data != nullptr && tensor->buffer != nullptr &&
        !ggml_backend_buffer_is_host(tensor->buffer) ? (const float *) tensor->data : nullptr;
}

static inline void ggml_cuda_nvfp4_set_tensor_header(
        const ggml_tensor * tensor, block_nvfp4_blackwell_tensor * dst,
        int64_t ne1, int64_t nplanes) {
    float weight_scale = 1.0f;
    float input_scale  = 1.0f;
    const ggml_tensor * weight_scale_t = tensor != nullptr ? tensor->src[0] : nullptr;
    const ggml_tensor * input_scale_t  = tensor != nullptr ? tensor->src[1] : nullptr;

    if (ggml_cuda_nvfp4_get_scalar_f32(weight_scale_t, &weight_scale) && !(weight_scale > 0.0f)) {
        weight_scale = 1.0f;
    }
    if (ggml_cuda_nvfp4_get_scalar_f32(input_scale_t, &input_scale) && !(input_scale > 0.0f)) {
        input_scale = 1.0f;
    }

    dst->weight_scale  = weight_scale;
    dst->input_scale   = input_scale;
    dst->weight_scales = ggml_cuda_nvfp4_scale_ptr(weight_scale_t);
    dst->input_scales  = ggml_cuda_nvfp4_scale_ptr(input_scale_t);
    GGML_UNUSED(ne1);
    GGML_UNUSED(nplanes);
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

static inline void ggml_cuda_repack_tiles_nvfp4(int64_t ne0, int64_t nrows, const void * src, void * dst) {
    GGML_ASSERT(ne0 % QK_NVFP4 == 0);

    const int64_t src_blocks_per_row = (ne0 + QK_NVFP4 - 1) / QK_NVFP4;
    const int64_t dst_blocks_per_row = ggml_cuda_nvfp4_blocks_per_row(ne0);
    const int64_t tile_rows = (nrows + 15) / 16;
    const size_t src_row_size = ggml_row_size(GGML_TYPE_NVFP4, ne0);

    const uint8_t * src_bytes = (const uint8_t *) src;
    block_nvfp4_blackwell * dst_blocks = (block_nvfp4_blackwell *) dst;

    for (int64_t tile_row = 0; tile_row < tile_rows; ++tile_row) {
        const int64_t row0 = tile_row * 16;
        const int rows_in_tile = (int) ((row0 + 16 <= nrows) ? 16 : (nrows - row0));

        for (int64_t block_col = 0; block_col < dst_blocks_per_row; ++block_col) {
            const int64_t src_block0 = block_col * 4;
            const int frags_in_block = (int) ((src_block0 + 4 <= src_blocks_per_row) ? 4 : (src_blocks_per_row - src_block0));

            block_nvfp4_blackwell & out = dst_blocks[tile_row * dst_blocks_per_row + block_col];
            if (rows_in_tile != 16 || frags_in_block != 4) {
                memset(&out, 0, sizeof(out));
            }

            for (int row_in_tile = 0; row_in_tile < rows_in_tile; ++row_in_tile) {
                const int64_t row = row0 + row_in_tile;
                const block_nvfp4 * src_row = (const block_nvfp4 *) (src_bytes + row * src_row_size);
                const int lane_base = (row_in_tile & 7) * 4;
                const int row_half = row_in_tile >> 3;
                const int scale_lane = lane_base + row_half;

                for (int frag = 0; frag < frags_in_block; ++frag) {
                    const block_nvfp4 & in = src_row[src_block0 + frag];
                    block_nvfp4_blackwell_frag & tile = out.tiles[frag];

                    const uint8_t * p0 = in.qs +  0;
                    const uint8_t * p1 = in.qs +  8;
                    const uint8_t * p2 = in.qs + 16;
                    const uint8_t * p3 = in.qs + 24;
                    tile.regs[lane_base + 0][row_half + 0] = ggml_cuda_bw_pack8(p0, 0);
                    tile.regs[lane_base + 1][row_half + 0] = ggml_cuda_bw_pack8(p0, 4);
                    tile.regs[lane_base + 2][row_half + 0] = ggml_cuda_bw_pack8(p1, 0);
                    tile.regs[lane_base + 3][row_half + 0] = ggml_cuda_bw_pack8(p1, 4);
                    tile.regs[lane_base + 0][row_half + 2] = ggml_cuda_bw_pack8(p2, 0);
                    tile.regs[lane_base + 1][row_half + 2] = ggml_cuda_bw_pack8(p2, 4);
                    tile.regs[lane_base + 2][row_half + 2] = ggml_cuda_bw_pack8(p3, 0);
                    tile.regs[lane_base + 3][row_half + 2] = ggml_cuda_bw_pack8(p3, 4);

                    uint32_t d = 0;
                    memcpy(&d, in.d, sizeof(d));
                    tile.scales_u32[scale_lane + 0] = d;
                    tile.scales_u32[scale_lane + 2] = d;
                }
            }
        }
    }
}

static inline void ggml_cuda_repack_tensor_nvfp4(const ggml_tensor * tensor, const void * src, void * dst) {
    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];
    const int64_t nplanes = tensor->ne[2] * tensor->ne[3];
    const size_t src_plane_size = ggml_row_size(GGML_TYPE_NVFP4, ne0) * ne1;
    const size_t dst_plane_size = ggml_cuda_nvfp4_plane_size(ne0, ne1);
    block_nvfp4_blackwell_tensor * dst_tensor = (block_nvfp4_blackwell_tensor *) dst;

    ggml_cuda_nvfp4_set_tensor_header(tensor, dst_tensor, ne1, nplanes);
    char * dst_tiles = (char *) dst_tensor->tiles;

    for (int64_t plane = 0; plane < nplanes; ++plane) {
        ggml_cuda_repack_tiles_nvfp4(ne0, ne1,
                (const char *) src + plane * src_plane_size,
                dst_tiles + plane * dst_plane_size);
    }
}

static inline void ggml_cuda_unpack_tiles_nvfp4(int64_t ne0, int64_t nrows, const void * src, void * dst) {
    GGML_ASSERT(ne0 % QK_NVFP4 == 0);

    const int64_t src_blocks_per_row = ggml_cuda_nvfp4_blocks_per_row(ne0);
    const int64_t dst_blocks_per_row = (ne0 + QK_NVFP4 - 1) / QK_NVFP4;
    const int64_t tile_rows = (nrows + 15) / 16;
    const size_t dst_row_size = ggml_row_size(GGML_TYPE_NVFP4, ne0);

    const block_nvfp4_blackwell * src_blocks = (const block_nvfp4_blackwell *) src;
    uint8_t * dst_bytes = (uint8_t *) dst;

    for (int64_t tile_row = 0; tile_row < tile_rows; ++tile_row) {
        const int64_t row0 = tile_row * 16;
        const int rows_in_tile = (int) ((row0 + 16 <= nrows) ? 16 : (nrows - row0));

        for (int64_t block_col = 0; block_col < src_blocks_per_row; ++block_col) {
            const int64_t dst_block0 = block_col * 4;
            const int frags_in_block = (int) ((dst_block0 + 4 <= dst_blocks_per_row) ? 4 : (dst_blocks_per_row - dst_block0));
            const block_nvfp4_blackwell & in = src_blocks[tile_row * src_blocks_per_row + block_col];

            for (int row_in_tile = 0; row_in_tile < rows_in_tile; ++row_in_tile) {
                const int64_t row = row0 + row_in_tile;
                block_nvfp4 * dst_row = (block_nvfp4 *) (dst_bytes + row * dst_row_size);
                const int lane_base = (row_in_tile & 7) * 4;
                const int row_half = row_in_tile >> 3;
                const int scale_lane = lane_base + row_half;

                for (int frag = 0; frag < frags_in_block; ++frag) {
                    const block_nvfp4_blackwell_frag & tile = in.tiles[frag];
                    block_nvfp4 & out = dst_row[dst_block0 + frag];

                    ggml_cuda_bw_unpack8(tile.regs[lane_base + 0][row_half + 0], tile.regs[lane_base + 1][row_half + 0], out.qs +  0);
                    ggml_cuda_bw_unpack8(tile.regs[lane_base + 2][row_half + 0], tile.regs[lane_base + 3][row_half + 0], out.qs +  8);
                    ggml_cuda_bw_unpack8(tile.regs[lane_base + 0][row_half + 2], tile.regs[lane_base + 1][row_half + 2], out.qs + 16);
                    ggml_cuda_bw_unpack8(tile.regs[lane_base + 2][row_half + 2], tile.regs[lane_base + 3][row_half + 2], out.qs + 24);

                    uint32_t d = tile.scales_u32[scale_lane];
                    memcpy(out.d, &d, sizeof(out.d));
                }
            }
        }
    }
}

static inline void ggml_cuda_unpack_tensor_nvfp4(const ggml_tensor * tensor, const void * src, void * dst) {
    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];
    const int64_t nplanes = tensor->ne[2] * tensor->ne[3];
    const size_t src_plane_size = ggml_cuda_nvfp4_plane_size(ne0, ne1);
    const size_t dst_plane_size = ggml_row_size(GGML_TYPE_NVFP4, ne0) * ne1;
    const block_nvfp4_blackwell_tensor * src_tensor = (const block_nvfp4_blackwell_tensor *) src;

    for (int64_t plane = 0; plane < nplanes; ++plane) {
        ggml_cuda_unpack_tiles_nvfp4(ne0, ne1,
                (const char *) src_tensor->tiles + plane * src_plane_size,
                (char *) dst + plane * dst_plane_size);
    }
}
