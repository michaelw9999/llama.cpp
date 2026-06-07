from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("CohereForCausalLM")
class CommandR2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.COMMAND_R

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # max_position_embeddings = 8192 in config.json but model was actually
        # trained on 128k context length
        # aya-23 models don't have model_max_length specified
        self.hparams["max_position_embeddings"] = self.find_hparam(["model_max_length", "max_position_embeddings"])

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_logit_scale(self.hparams["logit_scale"])
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)


@ModelBase.register("Cohere2ForCausalLM")
class Cohere2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.COHERE2

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        self.gguf_writer.add_logit_scale(self.hparams["logit_scale"])
        self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])
        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])

        rotary_pct = self.hparams["rotary_pct"]
        hidden_size = self.hparams["hidden_size"]
        num_attention_heads = self.hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(int(rotary_pct * (hidden_size // num_attention_heads)))
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Cohere2 runtime in llama.cpp expects no bias tensors;
        # the actual weight only contains 0-value tensors as bias, we can skip them
        if name.endswith(".bias"):
            if torch.any(data_torch != 0):
                raise ValueError(f"Bias tensor {name!r} is not zero.")
            logger.debug(f"Skipping bias tensor {name!r} for Cohere2 conversion.")
            return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Cohere2MoeForCausalLM")
class Cohere2MoeModel(TextModel):
    model_arch = gguf.MODEL_ARCH.COHERE2_MOE

    def get_vocab_base_pre(self, tokenizer) -> str:
        return "cohere2-moe"

    def _set_vocab_gpt2(self) -> None:
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        chat_template_jinja = self.dir_model / "chat_template.jinja"
        if chat_template_jinja.is_file():
            with open(chat_template_jinja, "r", encoding="utf-8") as f:
                special_vocab.chat_template = f.read()
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        hparams = self.hparams
        expert_intermediate_size = hparams["intermediate_size"]
        hparams["intermediate_size"] = hparams["prefix_dense_intermediate_size"]
        try:
            super().set_gguf_parameters()
        finally:
            hparams["intermediate_size"] = expert_intermediate_size

        self.gguf_writer.add_logit_scale(hparams["logit_scale"])
        self.gguf_writer.add_sliding_window(hparams["sliding_window"])
        self.gguf_writer.add_sliding_window_pattern([t == "sliding_attention" for t in hparams["layer_types"]])
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_expert_feed_forward_length(expert_intermediate_size)
        self.gguf_writer.add_leading_dense_block_count(hparams["first_k_dense_replace"])
        self.gguf_writer.add_expert_weights_norm(hparams.get("norm_topk_prob", False))
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        self.gguf_writer.add_rope_dimension_count(hparams["head_dim"])
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name == "lm_head.weight":
            logger.info("Skipping tied output layer 'lm_head.weight'")
            return

        if "mlp.experts" in name:
            n_experts = self.hparams["num_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)
                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    yield from super().modify_tensors(data_torch, merged_name, bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")
