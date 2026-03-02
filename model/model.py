import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
import math
from typing import Optional, Tuple, List, Union
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class MiniMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    freqs = torch.pow(
        rope_base, (1.0 / (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    )

    if rope_scaling is not None:
        original_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4.0),
            rope_scaling.get("beta_slow", 1.0),
        )

    # 寻找第一个波长大于 original_max 的频率索引，小于该索引的频率在训练时已经见过（转完了360°），大于的没见过
    if end / original_max > 1.0:
        corr_dim = next(
            (i for i in range(dim // 2) if 2 * math.pi / freqs[i] > original_max),
            dim // 2,
        )

        power = torch.arange(0, dim // 2, device=freqs.device).float() / max(
            dim // 2 - 1, 1
        )

        beta = beta_slow + (beta_fast - beta_slow) * power

        """
        [0, corr_dim): scale_i = (beta_i * factor - beta_i + 1)/ (beta_i * factor)
        [corr_dim, dim//2): scale_i = 1 / factor
        """
        scale = torch.where(
            torch.arange(dim // 2, device=freqs.device) < corr_dim,
            (beta * factor - beta + 1) / (beta * factor),
            1.0 / factor,
        )

        freqs = freqs * scale

    pos = torch.arange(end, device=freqs.device)
    freqs = torch.outer(pos, freqs)

    freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=None, unsqueeze_dim=1
) -> Tuple[torch.Tensor, torch.Tensor]:

    def rotate_half(x):
        return torch.cat(
            [-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]], dim=-1
        )

    # 考虑多头
    q_embed = q * cos.unsqueeze(unsqueeze_dim) + rotate_half(q) * sin.unsqueeze(
        unsqueeze_dim
    )
    k_embed = k * cos.unsqueeze(unsqueeze_dim) + rotate_half(k) * sin.unsqueeze(
        unsqueeze_dim
    )

    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, seq_len, num_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(bs, seq_len, num_heads, n_rep, head_dim)
        .reshape(bs, seq_len, num_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_rep = self.n_local_heads // self.local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.q_proj = nn.Linear(
            args.hidden_size, self.n_local_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.out_proj = nn.Linear(
            args.n_local_heads * self.head_dim, args.hidden_size, bias=False
        )

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attention
        )

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        bs, seq_len, _ = x.shape

        # 线性变换得到 q, k, v
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.view(bs, seq_len, self.n_local_heads, self.head_dim)
        k = k.view(bs, seq_len, self.local_kv_heads, self.head_dim)
        v = v.view(bs, seq_len, self.local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        # 应用旋转位置编码
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # kv_cache 处理
        if past_key_value is not None:
            # [bs, seq_len_cache, local_kv_heads, head_dim]
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)
        past_kv = (k, v) if use_cache else None

        q, k, v = (
            q.transpose(1, 2),
            repeat_kv(k, self.n_rep).transpose(1, 2),
            repeat_kv(v, self.n_rep).transpose(1, 2),
        )

        if (
            self.flash
            and (seq_len > 1)
            and (past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # [bs, nheads, seqlen, seqlen_kv]
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
            causal_mask = torch.full(
                [seq_len, seq_len], float("-inf"), device=x.device
            ).triu(diagonal=1)
            scores[..., -seq_len:] += causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            # [bs, seqlen_k] -> [bs, 1, 1, seqlen_k]
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # 1 表示 attend, 0 表示不 attend，乘以一个很大的负数（-1e9）使得 softmax 后接近于0
            extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
            scores = scores + extended_attention_mask

        scores = F.softmax(scores.float(), dim=-1).type_as(x)
        scores = self.attn_dropout(scores)
        output = scores @ v

        output = output.transpose(1, 2).reshape(bs, seq_len, -1)
        output = self.out_proj(output)
        output = self.resid_dropout(output)

        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size = args.hidden_size * 8 / 3
            args.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.dropout = nn.Dropout(args.dropout)
        self.activation = ACT2FN[args.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(
            self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))
        )


class MiniBlock(nn.Module):
    def __init__(self, layer_id: int, args: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.head_dim = self.hidden_size // args.num_attention_heads
        self.self_attn = Attention(args)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(self.hidden_size, eps=args.rms_norm_eps)
        self.post_layernorm = RMSNorm(self.hidden_size, eps=args.rms_norm_eps)
        self.mlp = FeedForward(args) if not args.use_moe else MOEFeedForward(args)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        residual = hidden_states

        # LayerNorm -> Self-Attention -> Dropout -> Residual
        hidden_states, presetn_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        # LayerNorm -> MLP -> Dropout -> Residual
        hidden_states = hidden_states + self.mlp(self.post_layernorm(hidden_states))
        return hidden_states, presetn_key_value


class MiniMindModel(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.dropout = nn.Dropout(args.dropout)
        self.layers = nn.ModuleList(
            [MiniBlock(i, args) for i in range(args.num_hidden_layers)]
        )
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs(
            args.hidden_size // args.num_attention_heads,
            end=args.max_position_embeddings,
            rope_base=args.rope_theta,
            rope_scaling=args.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        bs, seq_len = input_ids.shape
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        else:
            past_key_values = past_key_values

        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states)

        position_embedddings = (
            self.freqs_cos[start_pos : start_pos + seq_len],
            self.freqs_sin[start_pos : start_pos + seq_len],
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(
            zip(self.layers, past_key_values)
        ):
            hidden_states, present = layer(
                hidden_states,
                position_embedddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)

        return hidden_states, presents


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig):
        super().__init__(config)
        self.model = MiniMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ):
        hidden_states, past_kvs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_kvs,
            hidden_states=hidden_states,
        )
