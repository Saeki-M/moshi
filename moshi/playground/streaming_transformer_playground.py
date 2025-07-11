import typing as tp
from pathlib import Path
from typing import Optional, Unpack

import torch
import transformers
from transformers import LlamaModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from moshi.modules.rope import RotaryEmbedding
from moshi.modules.streaming import StreamingModule
from moshi.modules.transformer import StreamingTransformerLayer, _TransformerState


def process_llama_input(
    self: LlamaModel,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        print("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
        use_cache = False

    # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
    if not isinstance(past_key_values, (type(None), Cache)):
        raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **flash_attn_kwargs,
        )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)
    return hidden_states


class LlamaLM(StreamingModule[_TransformerState]):
    def __init__(
        self,
        d_model: int = None,
        num_heads: int = None,
        num_layers: int = None,
        dim_feedforward: int | list[int] = 2048,
        causal: bool = False,
        context: tp.Optional[int] = None,
        positional_embedding: str = "sin",
        max_period: float = 10_000,
        positional_scale: float = 1.0,
        betas: tp.Optional[tp.Tuple[float, float]] = None,
        layer_class: tp.Type[StreamingTransformerLayer] = StreamingTransformerLayer,
        quantize: bool = False,
        checkpointing: bool = False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        # assert d_model % num_heads == 0

        self.positional_embedding = positional_embedding
        self.max_period = max_period
        self.positional_scale = positional_scale
        self.betas = betas

        assert positional_embedding in {"sin", "rope", "sin_rope", "none"}
        self.rope: tp.Optional[RotaryEmbedding] = None
        if self.positional_embedding in {"rope", "sin_rope"}:
            self.rope = RotaryEmbedding(max_period=max_period)

        self.checkpointing = checkpointing

        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        llama_for_casual_lm = transformers.LlamaForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        self.llama_model = llama_for_casual_lm.model
        # self.rotary_emb = llama_model.model.rotary_emb
        # self.layers = llama_model.model.layers

        # TODO quantize llama model
        # for i in range(len(self.layers)):
        #     if quantize:
        #         # Quantizing layers one by one to avoid taking too much space during init.
        #         self.layers[i].to(device=device, dtype=dtype)
        #         replace_linear_with_qlinear(self.layers[i])

    def _init_streaming_state(self, batch_size: int) -> _TransformerState:
        device = next(self.parameters()).device
        return _TransformerState(batch_size, device, offsets=torch.zeros(batch_size, device=device, dtype=torch.long))

    def forward(self, x: torch.Tensor, *args, **kwargs):
        # this part should be replaced with llama model
        # however note this is also used for mimi and depth transformer
        B, T, C = x.shape

        dtype_input = x.dtype
        state = self._streaming_state
        if state is None:
            offsets = torch.zeros(1, dtype=torch.long, device=x.device)
        else:
            offsets = state.offsets

        # if self.positional_embedding in {"sin", "sin_rope"}:
        #     positions = torch.arange(T, device=x.device).view(1, -1, 1)
        #     positions = positions + offsets.view(-1, 1, 1)
        #     pos_emb = create_sin_embedding(
        #         positions, C, max_period=self.max_period, dtype=x.dtype
        #     )
        #     x = x + self.positional_scale * pos_emb

        # positions = torch.arange(T, device=x.device).view(1, -1, 1)
        # positions = positions + offsets.view(-1, 1, 1)
        # pos_emb = create_sin_embedding(
        #     positions, C, max_period=self.max_period, dtype=x.dtype
        # )

        # TODO: 学習を含め、checkpointingがtrueであることはあるか？
        # position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # for layer in self.layers:
        #     if self.checkpointing:
        #         y = torch_checkpoint(
        #             layer, x, *args, use_reentrant=False,
        #             determinism_check='none',
        #             preserve_rng_state=False,
        #             **kwargs)
        #         assert isinstance(y, torch.Tensor)
        #         x = y
        #     else:
        #         x = layer(
        #                 x,
        #                 # position_ids=positions,            # tensor of shape (B, T)
        #                 # attention_mask=None,               # or your real mask
        #                 # use_cache=self.checkpointing,      # if you want past_key_values
        #                 # output_attentions=False,
        #             )
        x = process_llama_input(self.llama_model, inputs_embeds=x)

        if state is not None:
            state.offsets[:] = torch.where(state.exec_mask, state.offsets + T, state.offsets)
        return x.to(dtype_input)


def test_input(device="cuda:0"):
    input_file = Path(__file__).parent / "./streaming_transformer_input.pt"
    input_data = torch.load(input_file, map_location=device)
    print(input_data)
    print(input_data.shape)

    model = LlamaLM()
    model.to(device)
    # print(model)

    print(model.forward(input_data))


if __name__ == "__main__":
    test_input()
