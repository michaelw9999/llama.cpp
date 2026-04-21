#include <ggml.h>

#include <cstdint>

int main() {
    struct ggml_init_params params = {
        /*.mem_size   =*/ 16u * 1024u * 1024u,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 3);
    ggml_tensor * w_scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * x_scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_tensor * mm = ggml_mul_mat(ctx, a, b);
    ggml_mul_mat_add_derived_tensor(mm, ggml_create_derived_tensor(
            w_scale,
            GGML_NVFP4_TENSOR_SCALE,
            GGML_DERIVED_TENSOR_FLAG_OPTIONAL));
    ggml_mul_mat_add_derived_tensor(mm, ggml_create_derived_tensor(
            x_scale,
            GGML_NVFP4_INPUT_SCALE,
            GGML_DERIVED_TENSOR_FLAG_OPTIONAL));
    GGML_ASSERT(ggml_get_derived_tensor(mm, GGML_NVFP4_TENSOR_SCALE) == w_scale);
    GGML_ASSERT(ggml_get_derived_tensor(mm, GGML_NVFP4_INPUT_SCALE) == x_scale);
    GGML_ASSERT(ggml_get_derived_tensor_flags(mm, GGML_NVFP4_TENSOR_SCALE) == GGML_DERIVED_TENSOR_FLAG_OPTIONAL);

    ggml_tensor * as = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 4, 2);
    ggml_tensor * bi = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 2, 3);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 3);

    ggml_tensor * mmid = ggml_mul_mat_id(ctx, as, bi, ids);
    ggml_mul_mat_add_derived_tensor(mmid, ggml_create_derived_tensor(
            w_scale,
            GGML_NVFP4_TENSOR_SCALE,
            GGML_DERIVED_TENSOR_FLAG_OPTIONAL));
    GGML_ASSERT(ggml_get_derived_tensor(mmid, GGML_NVFP4_TENSOR_SCALE) == w_scale);
    GGML_ASSERT(ggml_get_derived_tensor(mmid, GGML_NVFP4_INPUT_SCALE) == nullptr);

    ggml_free(ctx);
    return 0;
}
