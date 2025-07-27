import typing as tp
from typing import Optional, Unpack

import torch
import transformers
from transformers import LlamaModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from moshi.modules.rope import RotaryEmbedding
from moshi.modules.streaming import StreamingModule
from moshi.modules.transformer import StreamingTransformerLayer, _TransformerState
from moshi.utils.quantize import replace_linear_with_qlinear

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
        self.rotary_emb = llama_for_casual_lm.model.rotary_emb
        self.layers = llama_for_casual_lm.model.layers
        self.norm = llama_for_casual_lm.model.norm

        # TODO: quantization not working
        for i in range(len(self.layers)):
            if quantize:
                # Quantizing layers one by one to avoid taking too much space during init.
                self.layers[i].to(device=device, dtype=dtype)
                replace_linear_with_qlinear(self.layers[i])

    def _init_streaming_state(self, batch_size: int) -> _TransformerState:
        device = next(self.parameters()).device
        return _TransformerState(batch_size, device, offsets=torch.zeros(batch_size, device=device, dtype=torch.long))

    def forward(self, x: torch.Tensor, *args, **kwargs):
        # this part should be replaced with llama model
        # however note this is also used for mimi and depth transformer
        B, T, C = x.shape

        dtype_input = x.dtype
        state = self._streaming_state

        #--- llama model ---
        output_attentions = False
        cache_position = None
        past_key_values = None
        position_ids = None

        if self.checkpointing and self.training:
            print("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False
        else:
            use_cache = True

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + x.shape[1], device=x.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # causal_mask = self._update_causal_mask(
        #     attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        # )  # at least for inference, always return None
        causal_mask = None

        hidden_states = x

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]

        x = self.norm(hidden_states)
        #--- end llama model ---

        if state is not None:
            state.offsets[:] = torch.where(state.exec_mask, state.offsets + T, state.offsets)
        return x.to(dtype_input)
