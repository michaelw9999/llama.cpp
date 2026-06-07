#include "models.h"

void llama_model_cohere2_moe::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
    ml.get_key(LLM_KV_LOGIT_SCALE,                 hparams.f_logit_scale);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);

    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }

    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    const auto res = ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer());
    if (!res) {
        uint32_t swa_period = 4;
        ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, swa_period, false);
        hparams.set_swa_pattern(swa_period);
    }

    hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);

    switch (hparams.n_layer()) {
        case 49: type = LLM_TYPE_30B_A3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_cohere2_moe::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output      = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    output_s    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "scale"),       { 1 }, TENSOR_NOT_REQUIRED);
    output_in_s = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "input_scale"), { 1 }, TENSOR_NOT_REQUIRED);

    if (n_expert == 0) {
        throw std::runtime_error("n_expert must be > 0 for Cohere2Moe");
    }
    if (n_expert_used == 0) {
        throw std::runtime_error("n_expert_used must be > 0 for Cohere2Moe");
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        if (static_cast<uint32_t>(i) < hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, 0);
        } else {
            const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff;

            layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), { n_embd, n_expert }, 0);
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), { n_embd, n_ff_exp, n_expert }, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), { n_ff_exp, n_embd, n_expert }, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), { n_embd, n_ff_exp, n_expert }, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_cohere2_moe::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_cohere2_moe::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    const float f_logit_scale = hparams.f_logit_scale;
    const auto gating_func = hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE
        ? LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID
        : (llama_expert_gating_func_type) hparams.expert_gating_func;

    ggml_tensor * cur;
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv_iswa();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto effective_scale = [this](ggml_tensor * s, ggml_tensor * in_s, const char * name, int il) {
        if (s == nullptr) {
            return in_s;
        }
        if (in_s == nullptr) {
            return s;
        }
        ggml_tensor * cur = ggml_mul(ctx0, s, in_s);
        cb(cur, name, il);
        return cur;
    };

    for (int il = 0; il < n_layer; ++il) {
        const bool is_swa = hparams.is_swa(il);
        const bool force_rope = static_cast<uint32_t>(il) < hparams.n_layer_dense_lead;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * ffn_inp = cur;

        {
            llama_layer layer = model.layers[il];
            layer.wq_s   = effective_scale(layer.wq_s,   layer.wq_in_s,   "attn_q_s",   il);
            layer.wk_s   = effective_scale(layer.wk_s,   layer.wk_in_s,   "attn_k_s",   il);
            layer.wv_s   = effective_scale(layer.wv_s,   layer.wv_in_s,   "attn_v_s",   il);
            layer.wqkv_s = effective_scale(layer.wqkv_s, layer.wqkv_in_s, "attn_qkv_s", il);
            layer.wo_s   = effective_scale(layer.wo_s,   layer.wo_in_s,   "attn_out_s", il);

            auto [Qcur, Kcur, Vcur] = build_qkv(layer, cur,
                    n_embd_head, n_head, n_head_kv, il);

            if (is_swa || force_rope) {
                ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    layer.wo, layer.wo_b, layer.wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                    1.0f / sqrtf(float(n_embd_head)), il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur     = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpL    = ggml_get_rows(ctx0, inpL, inp_out_ids);
            ffn_inp = ggml_get_rows(ctx0, ffn_inp, inp_out_ids);
        }

        ggml_tensor * attn_out = cur;

        const auto & layer = model.layers[il];

        if (model.layers[il].ffn_gate_inp == nullptr) {
            cur = build_ffn(ffn_inp,
                    layer.ffn_up,   nullptr, effective_scale(layer.ffn_up_s, layer.ffn_up_in_s, "ffn_up_s", il),
                    layer.ffn_gate, nullptr, effective_scale(layer.ffn_gate_s, layer.ffn_gate_in_s, "ffn_gate_s", il),
                    layer.ffn_down, nullptr, effective_scale(layer.ffn_down_s, layer.ffn_down_in_s, "ffn_down_s", il),
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            ggml_tensor * moe_logits = build_lora_mm(layer.ffn_gate_inp, ffn_inp,
                    effective_scale(layer.ffn_gate_inp_s, layer.ffn_gate_inp_in_s, "ffn_gate_inp_s", il));
            cb(moe_logits, "ffn_moe_logits", il);

            cur = build_moe_ffn(ffn_inp,
                    nullptr,
                    layer.ffn_up_exps,
                    layer.ffn_gate_exps,
                    layer.ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    gating_func,
                    il,
                    moe_logits, nullptr,
                    effective_scale(layer.ffn_up_exps_s, layer.ffn_up_exps_in_s, "ffn_moe_up_s", il),
                    effective_scale(layer.ffn_gate_exps_s, layer.ffn_gate_exps_in_s, "ffn_moe_gate_s", il),
                    effective_scale(layer.ffn_down_exps_s, layer.ffn_down_exps_in_s, "ffn_moe_down_s", il));
            cb(cur, "ffn_moe_out", il);
        }

        cur = ggml_add(ctx0, cur, inpL);
        cur = ggml_add(ctx0, cur, attn_out);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, effective_scale(model.output_s, model.output_in_s, "output_s", -1));

    if (f_logit_scale) {
        cur = ggml_scale(ctx0, cur, f_logit_scale);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
