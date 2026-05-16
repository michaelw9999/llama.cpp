#include "mmvq.cuh"
#include "mma.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cstdint>
#include <type_traits>

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

#if defined(BLACKWELL_MMA_AVAILABLE)
static __device__ __forceinline__ float ggml_cuda_mmvq_apply_glu(
        const float result, const float gate_value, const ggml_glu_op glu_op) {
    switch (glu_op) {
        case GGML_GLU_OP_SWIGLU:
            return result * ggml_cuda_op_silu_single(gate_value);
        case GGML_GLU_OP_GEGLU:
            return result * ggml_cuda_op_gelu_single(gate_value);
        case GGML_GLU_OP_SWIGLU_OAI:
            return ggml_cuda_op_swiglu_oai_single(gate_value, result);
        default:
            return result * gate_value;
    }
}
#endif // defined(BLACKWELL_MMA_AVAILABLE)

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q2_0:    return vec_dot_q2_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_0:    return VDR_Q2_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_MXFP6_E2M3:   return VDR_MXFP6_E2M3_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4,
    MMVQ_PARAMETERS_GB10
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2) || defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    return MMVQ_PARAMETERS_GB10;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_DGX_SPARK) {
        return MMVQ_PARAMETERS_GB10;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_MXFP6_E2M3:   return 4;
        case GGML_TYPE_NVFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_MXFP6_E2M3:   return 7;
        case GGML_TYPE_NVFP4:   return 8;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_MXFP6_E2M3:   return 5;
        case GGML_TYPE_NVFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

template <ggml_type type>
static constexpr bool mmvq_small_k_turing_plus_allowed() {
    return type != GGML_TYPE_MXFP6_E2M3;
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // NVIDIA: Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        if (cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    if (!ggml_is_quantized(type)) {
        return false;
    }
    // k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner.
    // Only list quant-types MMQ supports, others would fall back to cuBLAS.
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ADA_LOVELACE) {
        switch (type) { // tuned on RTX 4090
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_BLACKWELL) {
        switch (type) { // tuned on RTX 5090
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 5;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_DGX_SPARK) {
        switch (type) { // tuned on DGX Spark GB10
            case GGML_TYPE_Q2_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        if (GGML_CUDA_CC_IS_CDNA1(cc)) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q5_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q8_0:
                    return ne11 <= 6;
                case GGML_TYPE_Q2_K:
                    return ne11 <= 4;
                case GGML_TYPE_Q3_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q4_K:
                    return ne11 <= 2;
                case GGML_TYPE_Q5_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q6_K:
                    return ne11 <= 4;
                case GGML_TYPE_IQ1_S:
                    return ne11 <= 5;
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_XS:
                    return ne11 <= 6;
                default:
                    return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        switch (type) { // tuned for CDNA2
            case GGML_TYPE_Q2_K:
                return ne11 <= 5;
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 3;
            case GGML_TYPE_Q6_K:
                return ne11 <= 5;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    return ne11 <= MMVQ_MAX_BATCH_SIZE;
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
    return MMVQ_MAX_BATCH_SIZE;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                    return 2;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_GB10) {
        const int generic = calc_nwarps(type, ncols_dst, MMVQ_PARAMETERS_GENERIC);
        // Only worth the wider block when it actually retires the K loop in half the trips (Observation)
        if (ncols_dst == 1 && !small_k && halve_iters) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 2 * generic;
                default:
                    break;
            }
        }
        return generic;
    }
    return 1;
}

static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GB10) {
        switch (ncols_dst) {
            case 1:
                return small_k ? nwarps : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

template <ggml_type type>
static __device__ __forceinline__ int get_mmvq_kbx(
        const uint32_t sample_x, const uint32_t channel_x, const uint32_t row,
        const uint32_t stride_sample_x, const uint32_t stride_channel_x,
        const uint32_t stride_row_x, const int kbx) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    if constexpr (type == GGML_TYPE_MXFP6_E2M3) {
        const uint32_t block_rel =
            sample_x*stride_sample_x + channel_x*stride_channel_x + (row / MXFP6_TILE_ROWS)*stride_row_x + kbx;
        return int(((row & (MXFP6_TILE_ROWS - 1)) << 28) | block_rel);
    } else
#endif // defined(BLACKWELL_MMA_AVAILABLE)
    {
        return int(sample_x*stride_sample_x + channel_x*stride_channel_x + row*stride_row_x + kbx);
    }
}

template <ggml_type type>
static __device__ __forceinline__ float vec_dot_mmvq(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y,
        const int kbx, const int kqs, const uint32_t channel_x) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    if constexpr (type == GGML_TYPE_MXFP6_E2M3) {
        return vec_dot_mxfp6_e2m3_q8_1(vx, y, kbx, kqs, channel_x);
    } else
#endif // defined(BLACKWELL_MMA_AVAILABLE)
    {
        GGML_UNUSED(channel_x);
        constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);
        return vec_dot_q_cuda(vx, y, kbx, kqs);
    }
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters)*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {

    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_k = qk / QK8_1;

    const int tid = warp_size*threadIdx.y + threadIdx.x;
    const int row0 = rows_per_cuda_block*blockIdx.x;
    int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    ggml_cuda_pdl_sync();
    channel_x  = ncols_dst == 1 && ids ? ids[channel_dst]                     : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op active_glu;
    float glu_limit = 0.0f;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        glu_limit     = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }
    }
    // Keep the no-fusion instantiation small; dense TG1 uses this generic path.
    [[maybe_unused]] float x_biases[has_fusion ? ncols_dst : 1]    = { 0.0f };
    [[maybe_unused]] float gate_biases[has_fusion ? ncols_dst : 1] = { 0.0f };
    [[maybe_unused]] float x_scales = 1.0f;
    [[maybe_unused]] float gate_scales = 1.0f;
    if constexpr (has_fusion) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        const bool active_row_thread = threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x);
        if (use_bias) {
            x_bias = x_bias + sample_dst*stride_sample_dst + channel_bias*stride_channel_dst + row0;
            if (active_row_thread) {
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
        }
        if (use_gate_bias) {
            gate_bias = gate_bias + sample_dst*stride_sample_dst + channel_bias*stride_channel_dst + row0;
            if (active_row_thread) {
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
        }
        if (active_row_thread) {
            if constexpr (type == GGML_TYPE_NVFP4) {
                if (use_scale) {
                    x_scales = x_scale[ids ? channel_x : 0];
                }
                if (use_gate_scale) {
                    gate_scales = gate_scale[ids ? channel_x : 0];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    // Avoid gate scratch in the no-fusion instantiation; it costs registers/shared memory.
    [[maybe_unused]] float tmp_gate[has_fusion ? ncols_dst : 1][has_fusion ? rows_per_cuda_block : 1] = {{0.0f}};

    const block_q8_1 * y_q8 = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_begin = tid / (qi/vdr);
    const int kbx_step  = blocks_per_iter;

    for (int kbx = kbx_begin; kbx < blocks_per_row_x; kbx += kbx_step) {
        const int kby = kbx * blocks_per_k;
        const int kqs_base = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                if (rows_per_cuda_block != 1 && uint32_t(row0 + i) >= nrows_x) {
                    continue;
                }
                const block_q8_1 * y_ptr_q8 = &y_q8[j*stride_col_y + kby];
                const uint32_t row = row0 + i;
                const int kbx_q = get_mmvq_kbx<type>(
                    sample_x, channel_x, row, stride_sample_x, stride_channel_x, stride_row_x, kbx);
                tmp[j][i] += vec_dot_mmvq<type>(vx, y_ptr_q8, kbx_q, kqs_base, channel_x);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_mmvq<type>(vgate, y_ptr_q8, kbx_q, kqs_base, channel_x);
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][has_fusion ? ncols_dst : 1][has_fusion ? rows_per_cuda_block : 1][warp_size];
    if constexpr (!has_fusion) {
        (void) tmp_shared_gate;
    } else if (!use_gate) {
        (void) tmp_shared_gate;
    }

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }

            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < nrows_x)) {
                float result = tmp[j][i];
                if constexpr (has_fusion) {
                    if constexpr (type == GGML_TYPE_NVFP4) {
                        result *= x_scales;
                    }
                    result += x_biases[j];
                    if (use_gate) {
                        float gate_value = tmp_gate[j][i];
                        if constexpr (type == GGML_TYPE_NVFP4) {
                            gate_value *= gate_scales;
                        }
                        gate_value += gate_biases[j];
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI:
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                break;
                            case GGML_GLU_OP_SWIGLU_CLAMP:
                                result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                }
                dst[j*stride_col_dst + i] = result;
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, active_glu, glu_limit, gate_bias, x_bias, x_scale, gate_scale, tmp_gate);
    }
    if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
    }
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    ggml_cuda_pdl_sync();
    const uint32_t channel_x = ids[channel_dst + token_idx * ids_stride];
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y_q8 = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
            for (int i = 0; i < c_rows_per_block; ++i) {
                if (c_rows_per_block != 1 && uint32_t(row0 + i) >= nrows_x) {
                    continue;
                }
            const uint32_t row = row0 + i;
            const int kbx_q = get_mmvq_kbx<type>(0, channel_x, row, 0, stride_channel_x, stride_row_x, kbx);
            tmp[i] += vec_dot_mmvq<type>(vx, &y_q8[kby], kbx_q, kqs, channel_x);
        }
    }

    ggml_cuda_pdl_lc();

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = tmp[threadIdx.x];
    }
}

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false, const bool halve_iters = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    const int rpb = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters>, launch_params,
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
}

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
            const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
            const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream) {

    constexpr int rows_per_block = 2; // 2 gives best perf based on tuning
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, nchannels_dst);
    const dim3 block_dims(warp_size, ncols_dst);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block>, launch_params,
        vx, vy, ids, dst, ncols_x, nchannels_y, nrows_x,
        stride_row_x, stride_col_y, stride_col_dst,
        stride_channel_x, stride_channel_y, stride_channel_dst,
        ncols_dst, ids_stride);
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_ids = ids != nullptr;

    // How the K loop divides up at the baseline block width, both decisions below use these.
    constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
    constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr                   = get_vdr_mmvq(type);
    const int     blocks_per_row_x      = ncols_x / qk;
    const int     blocks_per_iter_1warp = vdr * warp_size / qi;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        const int  nwarps = calc_nwarps(type, c_ncols_dst, table_id);
        bool       use    = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

        constexpr std::array<ggml_type, 2> iq_slow_turing = {
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S,
        };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = {
            GGML_TYPE_IQ3_S,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
        };

        const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

        if (is_nvidia_turing_plus) {
            if (ncols_dst == 1 &&
                    std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
            use = use && mmvq_small_k_turing_plus_allowed<type>();
        } else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
                (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
                GGML_CUDA_CC_IS_RDNA(cc)) {
            use = false;
        }

        return use;
    };

    // Whether doubling nwarps pays off on the ncols_dst == 1 path, where K sets the K loop trip count.
    const auto should_halve_iters = [&] {
        if (table_id != MMVQ_PARAMETERS_GB10) {
            return false;
        }

        // Expert rows are gathered per token, so a wider block adds reduction work without reuse.
        if (has_ids) {
            return false;
        }

        const int blocks_per_iter = calc_nwarps(type, 1, table_id) * blocks_per_iter_1warp;
        const int iters           = (blocks_per_row_x + blocks_per_iter - 1) /  blocks_per_iter;
        const int iters_wide      = (blocks_per_row_x + blocks_per_iter * 2 - 1) / (blocks_per_iter * 2);

        // An odd trip count leaves half the wider block idle for its last iteration, that tail is
        // only affordable once the loop is long enough to dilute it to an eighth of the work (observation).
        const int idle = iters_wide * 2 - iters;

        return idle * 8 <= iters_wide * 2;
    };

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel
            mul_mat_vec_q_moe_launch<type>(
                vx, vy, ids, dst, ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
        return;
    }

    switch (ncols_dst) {
        case 1: {
            // static, else MSVC lambda capture breaks the constexpr uses below
            static constexpr int c_ncols_dst = 1;

            // Tag types keep the flags compile-time, so __launch_bounds__ matches what is launched.
            const auto launch = [&](auto small_k_tag, auto halve_iters_tag) {
                constexpr bool c_small_k = decltype(small_k_tag)::value;
                // Types the table does not promote would compile a second, identical kernel.
                constexpr bool c_promoted =
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, true) !=
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, false);

                constexpr bool c_halve_iters = decltype(halve_iters_tag)::value && c_promoted;

                const std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                              nsamples_dst, warp_size, table_id, c_small_k, c_halve_iters);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, c_small_k, c_halve_iters>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream);
            };

            if (should_use_small_k(c_ncols_dst)) {
                launch(std::true_type{},  std::false_type{});
            } else if (should_halve_iters()) {
                launch(std::false_type{}, std::true_type{});
            } else {
                launch(std::false_type{}, std::false_type{});
            }
        } break;
        case 2: {
            constexpr int c_ncols_dst = 2;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                     channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                     sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                         dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 3: {
            constexpr int c_ncols_dst = 3;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                     channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                     sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                         dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 4: {
            constexpr int c_ncols_dst = 4;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                     channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                     sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                         dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                     channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                     sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                         dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                     channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                     sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                         dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                     channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                     sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                         dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                     channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                     sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                         dims.first, dims.second, 0, ids_stride, stream);
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}
static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
            const int stride_row_x, const int stride_col_y, const int stride_col_dst,
            const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q1_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
#if defined(BLACKWELL_MMA_AVAILABLE)
        case GGML_TYPE_MXFP6_E2M3:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP6_E2M3>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 ids_stride, stream);
            break;
#endif // defined(BLACKWELL_MMA_AVAILABLE)
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
        }
    }

#if defined(BLACKWELL_MMA_AVAILABLE)
static __device__ __forceinline__ void load_mxfp6_e2m3_tileA(
        ggml_cuda_mma::tile<16, 8, int> & A, uint32_t & scaleA,
        const tile_mxfp6_frag & frag) {
    const int lane = threadIdx.x & 31;
    const uint32_t w0 = frag.lane[lane][0];
    const uint32_t w1 = frag.lane[lane][1];
    const uint32_t w2 = frag.lane[lane][2];
    int * ax = (int *) A.x;
    ax[0] = (int) ggml_cuda_mxfp6_e2m3_codes_to_lanes(w0);
    ax[1] = (int) ggml_cuda_mxfp6_e2m3_codes_to_lanes((w0 >> 24) | (w1 << 8));
    ax[2] = (int) ggml_cuda_mxfp6_e2m3_codes_to_lanes((w1 >> 16) | (w2 << 16));
    ax[3] = (int) ggml_cuda_mxfp6_e2m3_codes_to_lanes(w2 >> 8);
    scaleA = frag.scale[lane];
}

static __device__ __forceinline__ void load_mxfp6_e2m3_q8_to_fp8_tileB(
        ggml_cuda_mma::tile<8, 8, int> & B, const block_q8_1 * __restrict__ y,
        const int stride_col_y, const int ncols) {
    const int lane = int(threadIdx.x) & 31;
    const int col  = lane >> 2;
    const int tid  = lane & 3;
    int * bx = (int *) B.x;

    if (col >= ncols) {
        bx[0] = 0;
        bx[1] = 0;
        return;
    }

    const block_q8_1 & bq8 = y[col*stride_col_y];
    const float d = __low2float(bq8.ds);
    bx[0] = int(uint32_t(__nv_cvt_float2_to_fp8x2(make_float2(float(bq8.qs[4*tid + 0]) * d, float(bq8.qs[4*tid + 1]) * d), __NV_SATFINITE, __NV_E4M3)) |
                uint32_t(__nv_cvt_float2_to_fp8x2(make_float2(float(bq8.qs[4*tid + 2]) * d, float(bq8.qs[4*tid + 3]) * d), __NV_SATFINITE, __NV_E4M3)) << 16);
    bx[1] = int(uint32_t(__nv_cvt_float2_to_fp8x2(make_float2(float(bq8.qs[4*tid + 16]) * d, float(bq8.qs[4*tid + 17]) * d), __NV_SATFINITE, __NV_E4M3)) |
                uint32_t(__nv_cvt_float2_to_fp8x2(make_float2(float(bq8.qs[4*tid + 18]) * d, float(bq8.qs[4*tid + 19]) * d), __NV_SATFINITE, __NV_E4M3)) << 16);
}

static __device__ __forceinline__ void load_fp8_tileB_1col_mmvq(
        ggml_cuda_mma::tile<8, 8, int> & B, uint32_t & scaleB,
        const block_fp8 * __restrict__ y, const int frag_abs32) {
    const int lane = int(threadIdx.x) & 31;
    const int tid  = lane & 3;
    const int block_rel = frag_abs32 / QK_FP8_FRAGS;
    const int frag_idx  = frag_abs32 % QK_FP8_FRAGS;
    int * bx = (int *) B.x;
    uint32_t sc = 0x7Fu;

    if (lane < 4) {
        const block_fp8 & bfp8 = y[block_rel];
        bx[0] = int(ggml_cuda_fp8_get4_u8containers(bfp8, frag_idx, tid + 0));
        bx[1] = int(ggml_cuda_fp8_get4_u8containers(bfp8, frag_idx, tid + 4));
        if (lane == 0) {
            sc = bfp8.e[frag_idx];
        }
    } else {
        bx[0] = 0;
        bx[1] = 0;
    }

    scaleB = __shfl_sync(0xFFFFFFFFu, sc, 0);
}

static __device__ __forceinline__ uint8_t fp8_fastq_scale_code_from_amax_mmvq(const float amax) {
    if (!(amax > 0.0f) || !isfinite(amax)) {
        return 127;
    }
    const float scaled = amax * (1.0f / 448.0f);
    const uint32_t bits = __float_as_uint(scaled);
    const int exp_bits = int((bits >> 23) & 0xff);
    int code = 0;
    if (exp_bits != 0) {
        code = exp_bits + (int(bits & 0x007fffff) != 0);
    }
    return uint8_t(max(0, min(254, code)));
}

static __global__ void quantize_row_fp8_fast_mmvq(
        const float * __restrict__ x, block_fp8 * __restrict__ y,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2) {
    const int64_t i1 = blockIdx.x;
    const int64_t k_block = blockIdx.y;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;
    const int64_t blocks_per_col = ggml_cuda_fp8_blocks_per_row(ne0);
    if (k_block >= blocks_per_col) {
        return;
    }

    const int tid = int(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int64_t base_idx = i3*s03 + i2*s02 + i1*s01;
    const int64_t batch_offset = (int64_t) blockIdx.z * (ne1 * blocks_per_col);
    block_fp8 * yb = y + batch_offset + i1*blocks_per_col + k_block;

    if (warp >= QK_FP8_FRAGS) {
        return;
    }

    const int frag = warp;
    const int elem0 = 2*lane + 0;
    const int elem1 = 2*lane + 1;
    const int64_t i00 = k_block*QK_FP8 + frag*QK_FP8_SUB;
    const float v0_raw = (lane < 16 && i00 + elem0 < ne00) ? x[base_idx + i00 + elem0] : 0.0f;
    const float v1_raw = (lane < 16 && i00 + elem1 < ne00) ? x[base_idx + i00 + elem1] : 0.0f;
    const float v0 = isfinite(v0_raw) ? v0_raw : 0.0f;
    const float v1 = isfinite(v1_raw) ? v1_raw : 0.0f;
    const float lane_amax = lane < 16 ? fmaxf(fabsf(v0), fabsf(v1)) : 0.0f;
    const float amax = warp_reduce_max<32>(lane_amax);
    const uint8_t scale_code = fp8_fastq_scale_code_from_amax_mmvq(amax);
    const float scale = ggml_cuda_e8m0_to_fp32(scale_code);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

    if (lane == 0) {
        yb->e[frag] = scale_code;
        if (frag == 0) {
            yb->pad[0] = 0;
        }
    }

    if (lane < 16) {
        const uint32_t fp8x2 = uint32_t(__nv_cvt_float2_to_fp8x2(
            make_float2(v0 * inv_scale, v1 * inv_scale), __NV_SATFINITE, __NV_E4M3));
        const int word = lane >> 1;
        const uint32_t packed = fp8x2 | (__shfl_sync(0xFFFFFFFFu, fp8x2, lane | 1) << 16);
        if ((lane & 1) == 0) {
            ggml_cuda_fp8_set4_u8containers(*yb, frag, word, packed);
        }
    }
}

static void quantize_row_fp8_fast_mmvq_cuda(
        const float * x, void * vy, const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    const int64_t block_num_y = ggml_cuda_fp8_blocks_per_row(ne0);
    const dim3 num_blocks(ne1, block_num_y, ne2 * ne3);
    const dim3 block_size(QK_FP8_FRAGS * 32, 1, 1);
    quantize_row_fp8_fast_mmvq<<<num_blocks, block_size, 0, stream>>>(
        x, (block_fp8 *) vy, ne00, s01, s02, s03, ne0, ne1, ne2);
}

template <int warps_per_tile>
static __global__ __launch_bounds__(32*warps_per_tile, 1) void mul_mat_vec_mxfp6_q8_to_fp8_1col(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y,
        float * __restrict__ dst, const int blocks_per_row_x,
        const int stride_channel_x, const int stride_sample_x,
        const int stride_channel_y, const int stride_sample_y,
        const int stride_channel_dst, const int stride_sample_dst,
        const int channel_ratio, const int sample_ratio) {
    using namespace ggml_cuda_mma;
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;

    const int tile_row = blockIdx.x;
    const int channel_dst = blockIdx.y;
    const int sample_dst = blockIdx.z;
    const int channel_x = channel_dst / channel_ratio;
    const int sample_x = sample_dst / sample_ratio;

    const tensor_mxfp6 * tensor = (const tensor_mxfp6 *) vx;
    const tile_mxfp6_frag * x = (const tile_mxfp6_frag *) ((const char *) tensor + MXFP6_HEADER_OFFSET) +
        sample_x*stride_sample_x + channel_x*stride_channel_x + tile_row*blocks_per_row_x;
    const block_q8_1 * y_cur = y + sample_dst*stride_sample_y + channel_dst*stride_channel_y;
    float tensor_scale = tensor->weight_scales ? tensor->weight_scales[channel_x] : tensor->weight_scale;
    tensor_scale = tensor_scale > 0.0f ? tensor_scale : 1.0f;

    const int warp_id = threadIdx.y;
    tile_C C = {};
    for (int kbx = warp_id; kbx < blocks_per_row_x; kbx += warps_per_tile) {
        tile_A A;
        tile_B B;
        load_mxfp6_e2m3_q8_to_fp8_tileB(B, y_cur + kbx, 0, 1);

        uint32_t scaleA;
        load_mxfp6_e2m3_tileA(A, scaleA, x[kbx]);
        mma_block_scaled_mxfp6_e2m3_fp8_e4m3(C, A, B, scaleA, 0x7Fu);
    }

    __shared__ float partial[16][warps_per_tile];
    if ((threadIdx.x & 3) == 0) {
        const int row_part = threadIdx.x >> 2;
        partial[row_part + 0][warp_id] = C.x[0];
        partial[row_part + 8][warp_id] = C.x[2];
    }
    __syncthreads();

    if (warp_id == 0 && threadIdx.x < 16) {
        const int row_in_tile = threadIdx.x;
        const int row = tile_row * tile_C::I + row_in_tile;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < warps_per_tile; ++w) {
            sum += partial[row_in_tile][w];
        }
        dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row] = tensor_scale * sum;
    }
}

#define GGML_CUDA_MXFP6_DIRECT_FP8_MIN_BLOCKS(W) ((W) <= 16 ? 2 : 1)

static __device__ __forceinline__ void mma_block_scaled_mxfp6_e2m3_fp8_e4m3_1col(
        float & c0, float & c2,
        const ggml_cuda_mma::tile<16, 8, int> & A,
        const ggml_cuda_mma::tile<8, 8, int> & B,
        const uint32_t scaleA, const uint32_t scaleB) {
    const int * Axi = (const int *) A.x;
    const int * Bxi = (const int *) B.x;
    float c1 = 0.0f;
    float c3 = 0.0f;

    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m3.e4m3.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "{%10}, {0, 0}, {%11}, {0, 0};"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]),
          "r"(Bxi[0]), "r"(Bxi[1]), "r"(scaleA), "r"(scaleB));
}

template <int warps_per_tile>
static __global__ __launch_bounds__(32*warps_per_tile, GGML_CUDA_MXFP6_DIRECT_FP8_MIN_BLOCKS(warps_per_tile)) void mul_mat_vec_mxfp6_fp8_1col(
        const void * __restrict__ vx, const block_fp8 * __restrict__ y,
        float * __restrict__ dst, const int blocks_per_row_x,
        const int stride_channel_x, const int stride_sample_x,
        const int stride_channel_y, const int stride_sample_y,
        const int stride_channel_dst, const int stride_sample_dst,
        const int channel_ratio, const int sample_ratio) {
    using namespace ggml_cuda_mma;
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;

    const int tile_row = blockIdx.x;
    const int row_base = tile_row * tile_C::I;
    const int channel_dst = blockIdx.y;
    const int sample_dst = blockIdx.z;
    const int channel_x = channel_dst / channel_ratio;
    const int sample_x = sample_dst / sample_ratio;

    const tensor_mxfp6 * tensor = (const tensor_mxfp6 *) vx;
    const tile_mxfp6_frag * x = (const tile_mxfp6_frag *) ((const char *) tensor + MXFP6_HEADER_OFFSET) +
        sample_x*stride_sample_x + channel_x*stride_channel_x + tile_row*blocks_per_row_x;
    const block_fp8 * y_cur = y + sample_dst*stride_sample_y + channel_dst*stride_channel_y;

    const int warp_id = threadIdx.y;
    float c0 = 0.0f;
    float c2 = 0.0f;
    for (int kbx = warp_id; kbx < blocks_per_row_x; kbx += warps_per_tile) {
        tile_B B;
        uint32_t scaleB;
        load_fp8_tileB_1col_mmvq(B, scaleB, y_cur, kbx);

        tile_A A;
        uint32_t scaleA;
        load_mxfp6_e2m3_tileA(A, scaleA, x[kbx]);
        mma_block_scaled_mxfp6_e2m3_fp8_e4m3_1col(c0, c2, A, B, scaleA, scaleB);
    }

    __shared__ float partial[16][warps_per_tile];
    if ((threadIdx.x & 3) == 0) {
        const int row_part = threadIdx.x >> 2;
        partial[row_part + 0][warp_id] = c0;
        partial[row_part + 8][warp_id] = c2;
    }
    __syncthreads();

    if (warp_id == 0 && threadIdx.x < 16) {
        const int row_in_tile = threadIdx.x;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < warps_per_tile; ++w) {
            sum += partial[row_in_tile][w];
        }
        float tensor_scale = tensor->weight_scales ? tensor->weight_scales[channel_x] : tensor->weight_scale;
        tensor_scale = tensor_scale > 0.0f ? tensor_scale : 1.0f;
        dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row_base + row_in_tile] = tensor_scale * sum;
    }
}

template <int warps_per_tile, bool has_x_bias, bool has_gate_bias>
static __global__ __launch_bounds__(32*warps_per_tile, GGML_CUDA_MXFP6_DIRECT_FP8_MIN_BLOCKS(warps_per_tile)) void mul_mat_vec_mxfp6_fp8_gate_1col(
        const void * __restrict__ vx, const void * __restrict__ vgate, const block_fp8 * __restrict__ y,
        const float * __restrict__ x_bias, const float * __restrict__ gate_bias,
        const ggml_glu_op glu_op, float * __restrict__ dst, const int blocks_per_row_x,
        const int stride_channel_x, const int stride_sample_x,
        const int stride_channel_y, const int stride_sample_y,
        const int stride_channel_dst, const int stride_sample_dst,
        const int channel_ratio, const int sample_ratio) {
    using namespace ggml_cuda_mma;
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;

    const int tile_row = blockIdx.x;
    const int row_base = tile_row * tile_C::I;
    const int channel_dst = blockIdx.y;
    const int sample_dst = blockIdx.z;
    const int channel_x = channel_dst / channel_ratio;
    const int sample_x = sample_dst / sample_ratio;

    const tensor_mxfp6 * tensor_x = (const tensor_mxfp6 *) vx;
    const tensor_mxfp6 * tensor_gate = (const tensor_mxfp6 *) vgate;
    const tile_mxfp6_frag * x = (const tile_mxfp6_frag *) ((const char *) tensor_x + MXFP6_HEADER_OFFSET) +
        sample_x*stride_sample_x + channel_x*stride_channel_x + tile_row*blocks_per_row_x;
    const tile_mxfp6_frag * gate = (const tile_mxfp6_frag *) ((const char *) tensor_gate + MXFP6_HEADER_OFFSET) +
        sample_x*stride_sample_x + channel_x*stride_channel_x + tile_row*blocks_per_row_x;
    const block_fp8 * y_cur = y + sample_dst*stride_sample_y + channel_dst*stride_channel_y;

    const int warp_id = threadIdx.y;
    float c0 = 0.0f;
    float c2 = 0.0f;
    float gate_c0 = 0.0f;
    float gate_c2 = 0.0f;
    for (int kbx = warp_id; kbx < blocks_per_row_x; kbx += warps_per_tile) {
        tile_B B;
        uint32_t scaleB;
        load_fp8_tileB_1col_mmvq(B, scaleB, y_cur, kbx);

        {
            tile_A A;
            uint32_t scaleA;
            load_mxfp6_e2m3_tileA(A, scaleA, x[kbx]);
            mma_block_scaled_mxfp6_e2m3_fp8_e4m3_1col(c0, c2, A, B, scaleA, scaleB);
        }
        {
            tile_A A;
            uint32_t scaleA;
            load_mxfp6_e2m3_tileA(A, scaleA, gate[kbx]);
            mma_block_scaled_mxfp6_e2m3_fp8_e4m3_1col(gate_c0, gate_c2, A, B, scaleA, scaleB);
        }
    }

    __shared__ float partial[2][16][warps_per_tile];
    if ((threadIdx.x & 3) == 0) {
        const int row_part = threadIdx.x >> 2;
        partial[0][row_part + 0][warp_id] = c0;
        partial[0][row_part + 8][warp_id] = c2;
        partial[1][row_part + 0][warp_id] = gate_c0;
        partial[1][row_part + 8][warp_id] = gate_c2;
    }
    __syncthreads();

    if (warp_id == 0 && threadIdx.x < 16) {
        const int row_in_tile = threadIdx.x;
        const int row = row_base + row_in_tile;
        float sum = 0.0f;
        float gate_sum = 0.0f;
#pragma unroll
        for (int w = 0; w < warps_per_tile; ++w) {
            sum      += partial[0][row_in_tile][w];
            gate_sum += partial[1][row_in_tile][w];
        }

        float tensor_scale = tensor_x->weight_scales ? tensor_x->weight_scales[channel_x] : tensor_x->weight_scale;
        tensor_scale = tensor_scale > 0.0f ? tensor_scale : 1.0f;
        float gate_tensor_scale = tensor_gate->weight_scales ? tensor_gate->weight_scales[channel_x] : tensor_gate->weight_scale;
        gate_tensor_scale = gate_tensor_scale > 0.0f ? gate_tensor_scale : 1.0f;
        float result = tensor_scale * sum;
        float gate_value = gate_tensor_scale * gate_sum;
        if constexpr (has_x_bias) {
            result += x_bias[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row];
        }
        if constexpr (has_gate_bias) {
            gate_value += gate_bias[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row];
        }
        dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row] =
            ggml_cuda_mmvq_apply_glu(result, gate_value, glu_op);
    }
}

template <int warps_per_tile>
static __global__ __launch_bounds__(32*warps_per_tile, 1) void mul_mat_vec_mxfp6_q8_to_fp8_cols(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y,
        float * __restrict__ dst, const int blocks_per_row_x, const int stride_channel_x,
        const int stride_col_y, const int stride_channel_y, const int stride_sample_y,
        const int stride_col_dst, const int stride_channel_dst, const int stride_sample_dst,
        const int ncols_dst) {
    using namespace ggml_cuda_mma;
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;

    const int tile_row = blockIdx.x;
    const int channel_dst = blockIdx.y;
    const int sample_dst = blockIdx.z;

    const tensor_mxfp6 * tensor = (const tensor_mxfp6 *) vx;
    const tile_mxfp6_frag * x = (const tile_mxfp6_frag *) ((const char *) tensor + MXFP6_HEADER_OFFSET) +
        (sample_dst*int(gridDim.y) + channel_dst)*stride_channel_x + tile_row*blocks_per_row_x;
    const block_q8_1 * y_cur = y + sample_dst*stride_sample_y + channel_dst*stride_channel_y;
    float tensor_scale = tensor->weight_scales ? tensor->weight_scales[channel_dst] : tensor->weight_scale;
    tensor_scale = tensor_scale > 0.0f ? tensor_scale : 1.0f;

    const int warp_id = threadIdx.y;
    tile_C C = {};
    for (int kbx = warp_id; kbx < blocks_per_row_x; kbx += warps_per_tile) {
        tile_A A;
        tile_B B;
        load_mxfp6_e2m3_q8_to_fp8_tileB(B, y_cur + kbx, stride_col_y, ncols_dst);

        uint32_t scaleA;
        load_mxfp6_e2m3_tileA(A, scaleA, x[kbx]);
        mma_block_scaled_mxfp6_e2m3_fp8_e4m3(C, A, B, scaleA, 0x7Fu);
    }

    __shared__ float partial[16][8][warps_per_tile];
#pragma unroll
    for (int l = 0; l < tile_C::ne; ++l) {
        partial[tile_C::get_i(l)][tile_C::get_j(l)][warp_id] = C.x[l];
    }
    __syncthreads();

    if (warp_id == 0) {
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int row = tile_row * tile_C::I + tile_C::get_i(l);
            const int col = tile_C::get_j(l);
            if (col < ncols_dst) {
                float sum = 0.0f;
#pragma unroll
                for (int w = 0; w < warps_per_tile; ++w) {
                    sum += partial[row - tile_row * tile_C::I][col][w];
                }
                dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + col*stride_col_dst + row] =
                    tensor_scale * sum;
            }
        }
    }
}

template <int warps_per_tile>
static __global__ __launch_bounds__(32*warps_per_tile, 1) void mul_mat_vec_mxfp6_q8_to_fp8_moe(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y, const int32_t * __restrict__ ids,
        float * __restrict__ dst, const int blocks_per_row_x, const int stride_channel_x,
        const int stride_col_y, const int stride_channel_y, const int stride_col_dst,
        const int stride_channel_dst, const int ids_stride, const int nchannels_y) {
    using namespace ggml_cuda_mma;
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;

    const int tile_row = blockIdx.x;
    const int channel_dst = blockIdx.y;
    const int token_idx = blockIdx.z;
    const int channel_x = ids[channel_dst + token_idx*ids_stride];
    const int channel_y = channel_dst % nchannels_y;

    const tensor_mxfp6 * tensor = (const tensor_mxfp6 *) vx;
    const tile_mxfp6_frag * x = (const tile_mxfp6_frag *) ((const char *) tensor + MXFP6_HEADER_OFFSET) +
        channel_x*stride_channel_x + tile_row*blocks_per_row_x;
    const block_q8_1 * y_cur = y + channel_y*stride_channel_y + token_idx*stride_col_y;
    float tensor_scale = tensor->weight_scales ? tensor->weight_scales[channel_x] : tensor->weight_scale;
    tensor_scale = tensor_scale > 0.0f ? tensor_scale : 1.0f;

    const int warp_id = threadIdx.y;
    tile_C C = {};
    for (int kbx = warp_id; kbx < blocks_per_row_x; kbx += warps_per_tile) {
        tile_A A;
        tile_B B;
        load_mxfp6_e2m3_q8_to_fp8_tileB(B, y_cur + kbx, 0, 1);

        uint32_t scaleA;
        load_mxfp6_e2m3_tileA(A, scaleA, x[kbx]);
        mma_block_scaled_mxfp6_e2m3_fp8_e4m3(C, A, B, scaleA, 0x7Fu);
    }

    __shared__ float partial[16][warps_per_tile];
    if ((threadIdx.x & 3) == 0) {
        const int row_part = threadIdx.x >> 2;
        partial[row_part + 0][warp_id] = C.x[0];
        partial[row_part + 8][warp_id] = C.x[2];
    }
    __syncthreads();

    if (warp_id == 0 && threadIdx.x < 16) {
        const int row_in_tile = threadIdx.x;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < warps_per_tile; ++w) {
            sum += partial[row_in_tile][w];
        }
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + tile_row*tile_C::I + row_in_tile] =
            tensor_scale * sum;
    }
}
#endif // defined(BLACKWELL_MMA_AVAILABLE)

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        GGML_ASSERT( !ids || dst->ne[2] == 1);
        GGML_ASSERT(  ids || dst->ne[1] == 1);
        // Scale fusion is only allowed for NVFP4 currently as the cost of checking this at run-time in the prologue is
        // non-negligible for some models such as gpt-oss-20b
        GGML_ASSERT((fusion->x_scale == nullptr && fusion->gate_scale == nullptr) || src0->type == GGML_TYPE_NVFP4);

        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        if (fusion->x_scale) {
            GGML_ASSERT(fusion->x_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->x_scale));
            GGML_ASSERT(ggml_nelements(fusion->x_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.x_scale = fusion->x_scale->data;
        }
        if (fusion->gate_scale) {
            GGML_ASSERT(fusion->gate_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->gate_scale));
            GGML_ASSERT(ggml_nelements(fusion->gate_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.gate_scale = fusion->gate_scale->data;
        }
        fusion_local.glu_op = fusion->glu_op;
        fusion_local.glu_limit = fusion->glu_limit;
    }
    // If src0 is a temporary compute buffer, clear any potential padding.
    if (src0->type != GGML_TYPE_NVFP4 &&
            src0->type != GGML_TYPE_MXFP6_E2M3 &&
            ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_channel_dst = ids ? s1   : s2;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    int64_t stride_row_x = s01;
    int64_t stride_channel_x = s02;
    int64_t stride_sample_x = s03;
    if (src0->type == GGML_TYPE_MXFP6_E2M3) {
        GGML_ASSERT(src0->view_src == nullptr);
        GGML_ASSERT(ne00 % QK_MXFP6_E2M3 == 0);
        const int64_t blocks_per_row_x = ggml_cuda_mxfp6_e2m3_frags_per_row(ne00);
        const int64_t tiles_per_channel_x = (ne01 + MXFP6_TILE_ROWS - 1) / MXFP6_TILE_ROWS;
        stride_row_x = blocks_per_row_x;
        stride_channel_x = tiles_per_channel_x * blocks_per_row_x;
        stride_sample_x = (s03 / s02) * stride_channel_x;
    }

    const int64_t s11_src1 = src1->nb[1] / ts_src1;
    const int64_t s12_src1 = src1->nb[2] / ts_src1;
    const int64_t s13_src1 = src1->nb[3] / ts_src1;
    const void * src0_data = src0->data;
    const bool has_fusion = fusion_local.gate != nullptr || fusion_local.x_bias != nullptr || fusion_local.gate_bias != nullptr;
    const bool use_mxfp6_fp8_act =
#if defined(BLACKWELL_MMA_AVAILABLE)
        src0->type == GGML_TYPE_MXFP6_E2M3 &&
        !ids && ncols_dst == 1 && ne01 % MXFP6_TILE_ROWS == 0 &&
        ((fusion_local.gate == nullptr && fusion_local.x_bias == nullptr && fusion_local.gate_bias == nullptr) ||
         fusion_local.gate != nullptr);
#else
        false;
#endif // defined(BLACKWELL_MMA_AVAILABLE)

    const int64_t blocks_per_row_y = use_mxfp6_fp8_act ? ggml_cuda_fp8_blocks_per_row(ne10_padded) : ne10_padded / QK8_1;
    const size_t y_block_size = use_mxfp6_fp8_act ? sizeof(block_fp8) : sizeof(block_q8_1);
    const size_t nbytes_src1 = size_t(ne13*ne12 * ne11 * blocks_per_row_y * y_block_size);
    ggml_cuda_pool_alloc<char> src1_t(ctx.pool(), nbytes_src1);

    int64_t stride_col_y = ids ? ne11 * blocks_per_row_y : blocks_per_row_y;
    int64_t stride_channel_y = ids ? blocks_per_row_y : ne11 * blocks_per_row_y;
    int64_t stride_sample_y = ne12 * ne11 * blocks_per_row_y;

#if defined(BLACKWELL_MMA_AVAILABLE)
    if (use_mxfp6_fp8_act) {
        quantize_row_fp8_fast_mmvq_cuda(src1_d, src1_t.get(), ne10, s11_src1, s12_src1, s13_src1,
                ne10_padded, ne11, ne12, ne13, stream);
    } else
#endif // defined(BLACKWELL_MMA_AVAILABLE)
    {
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_t.get(), src0->type, ne10, s11_src1, s12_src1, s13_src1,
                ne10_padded, ne11, ne12, ne13, stream);
    }
    CUDA_CHECK(cudaGetLastError());

#if defined(BLACKWELL_MMA_AVAILABLE)
    if (src0->type == GGML_TYPE_MXFP6_E2M3 && ne01 % 16 == 0 &&
            (!has_fusion || (use_mxfp6_fp8_act && fusion_local.gate != nullptr))) {
        const int blocks_per_row_x = int(ggml_cuda_mxfp6_e2m3_frags_per_row(ne00));
        const int tile_rows_x = int((ne01 + MXFP6_TILE_ROWS - 1) / MXFP6_TILE_ROWS);
        const int stride_channel_x_tiles = tile_rows_x * blocks_per_row_x;
        constexpr int warps_per_tile = 16;
        const dim3 block_dims(32, warps_per_tile, 1);

        if (ids && ncols_dst > 1) {
            const dim3 block_nums(tile_rows_x, nchannels_dst, ncols_dst);
            mul_mat_vec_mxfp6_q8_to_fp8_moe<warps_per_tile><<<block_nums, block_dims, 0, stream>>>(
                src0_data, (const block_q8_1 *) src1_t.get(), ids_d, dst_d,
                blocks_per_row_x, stride_channel_x_tiles, int(stride_col_y), int(stride_channel_y),
                int(stride_col_dst), int(stride_channel_dst), int(ids_stride), int(nchannels_y));
            CUDA_CHECK(cudaGetLastError());
            return;
        }

        if (!ids && ncols_dst == 1) {
            const dim3 block_nums(tile_rows_x, nchannels_dst, ne3);
            const int channel_ratio_simple = int(nchannels_dst / ne02);
            const int sample_ratio_simple = int(ne3 / ne03);
            if (use_mxfp6_fp8_act) {
                if (fusion_local.gate != nullptr) {
                    if (fusion_local.x_bias != nullptr && fusion_local.gate_bias != nullptr) {
                        mul_mat_vec_mxfp6_fp8_gate_1col<warps_per_tile, true, true><<<block_nums, block_dims, 0, stream>>>(
                            src0_data, fusion_local.gate, (const block_fp8 *) src1_t.get(),
                            (const float *) fusion_local.x_bias, (const float *) fusion_local.gate_bias, fusion_local.glu_op, dst_d,
                            blocks_per_row_x, stride_channel_x_tiles, int(stride_sample_x),
                            int(stride_channel_y), int(stride_sample_y), int(stride_channel_dst), int(s3),
                            channel_ratio_simple, sample_ratio_simple);
                    } else if (fusion_local.x_bias != nullptr) {
                        mul_mat_vec_mxfp6_fp8_gate_1col<warps_per_tile, true, false><<<block_nums, block_dims, 0, stream>>>(
                            src0_data, fusion_local.gate, (const block_fp8 *) src1_t.get(),
                            (const float *) fusion_local.x_bias, nullptr, fusion_local.glu_op, dst_d,
                            blocks_per_row_x, stride_channel_x_tiles, int(stride_sample_x),
                            int(stride_channel_y), int(stride_sample_y), int(stride_channel_dst), int(s3),
                            channel_ratio_simple, sample_ratio_simple);
                    } else if (fusion_local.gate_bias != nullptr) {
                        mul_mat_vec_mxfp6_fp8_gate_1col<warps_per_tile, false, true><<<block_nums, block_dims, 0, stream>>>(
                            src0_data, fusion_local.gate, (const block_fp8 *) src1_t.get(),
                            nullptr, (const float *) fusion_local.gate_bias, fusion_local.glu_op, dst_d,
                            blocks_per_row_x, stride_channel_x_tiles, int(stride_sample_x),
                            int(stride_channel_y), int(stride_sample_y), int(stride_channel_dst), int(s3),
                            channel_ratio_simple, sample_ratio_simple);
                    } else {
                        mul_mat_vec_mxfp6_fp8_gate_1col<warps_per_tile, false, false><<<block_nums, block_dims, 0, stream>>>(
                            src0_data, fusion_local.gate, (const block_fp8 *) src1_t.get(),
                            nullptr, nullptr, fusion_local.glu_op, dst_d,
                            blocks_per_row_x, stride_channel_x_tiles, int(stride_sample_x),
                            int(stride_channel_y), int(stride_sample_y), int(stride_channel_dst), int(s3),
                            channel_ratio_simple, sample_ratio_simple);
                    }
                } else {
                    mul_mat_vec_mxfp6_fp8_1col<warps_per_tile><<<block_nums, block_dims, 0, stream>>>(
                        src0_data, (const block_fp8 *) src1_t.get(), dst_d,
                        blocks_per_row_x, stride_channel_x_tiles, int(stride_sample_x),
                        int(stride_channel_y), int(stride_sample_y), int(stride_channel_dst), int(s3),
                        channel_ratio_simple, sample_ratio_simple);
                }
            } else {
                mul_mat_vec_mxfp6_q8_to_fp8_1col<warps_per_tile><<<block_nums, block_dims, 0, stream>>>(
                    src0_data, (const block_q8_1 *) src1_t.get(), dst_d,
                    blocks_per_row_x, stride_channel_x_tiles, int(stride_sample_x),
                    int(stride_channel_y), int(stride_sample_y), int(stride_channel_dst), int(s3),
                    channel_ratio_simple, sample_ratio_simple);
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }

        if (!ids && ncols_dst <= 8 && nchannels_dst == ne02 && ne3 == ne03) {
            const dim3 block_nums(tile_rows_x, nchannels_dst, ne3);
            mul_mat_vec_mxfp6_q8_to_fp8_cols<warps_per_tile><<<block_nums, block_dims, 0, stream>>>(
                src0_data, (const block_q8_1 *) src1_t.get(), dst_d,
                blocks_per_row_x, stride_channel_x_tiles, int(stride_col_y), int(stride_channel_y), int(stride_sample_y),
                int(stride_col_dst), int(stride_channel_dst), int(s3), int(ncols_dst));
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
#endif // defined(BLACKWELL_MMA_AVAILABLE)

    mul_mat_vec_q_switch_type(
        src0_data, src0->type, src1_t.get(), ids_d, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     stride_row_x, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           stride_sample_x, stride_sample_y, s3,               ids_stride, stream);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    int stride_row_x = ne00 / ggml_blck_size(src0->type);
    int stride_col_y = src1_padded_row_size / QK8_1;

    if (src0->type == GGML_TYPE_MXFP6_E2M3 && src0->view_src == nullptr &&
            ne00 % QK_MXFP6_E2M3 == 0) {
        stride_row_x = ggml_cuda_mxfp6_e2m3_frags_per_row(ne00);
    }

    ggml_cuda_mm_fusion_args_device fusion_local{};
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}
