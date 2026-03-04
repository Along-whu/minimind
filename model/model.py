import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
import math
from typing import Optional, Tuple, List, Union
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.autograd.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity


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


# 专家模块
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # 双层 MLP：Linear→GELU→Linear
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),  # 比 ReLU 更平滑的激活函数
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)  # 前向传播


# MoE 核心模块 Switch-Transformer
class MoE(nn.Module):
    def __init__(
        self, input_dim, num_experts, top_k, expert_capacity, hidden_dim, output_dim
    ):
        super().__init__()
        self.num_experts = num_experts  # 专家数量：需根据任务复杂度调整（如简单任务 4-8 个，复杂任务 16-32 个）
        self.top_k = top_k  # 每个样本激活的专家数：核心稀疏参数，通常取 1-4（K=2 是兼顾效率与性能的常用值）
        self.expert_capacity = (
            expert_capacity  # 单个专家最大处理样本数：避免“热门专家”过载导致 OOM
        )

        # 路由门控网络：输入 x→输出各专家的匹配度（logits），维度为[batch_size, num_experts]
        self.gate = nn.Linear(
            input_dim, num_experts
        )  # 线性层是门控的极简实现，复杂场景可替换为 Transformer 层

        # 创建专家集合：用 nn.ModuleList 管理，支持自动参数注册与设备迁移
        self.experts = nn.ModuleList(
            [Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)]
        )

    def forward(self, x):
        batch_size, input_dim = x.shape
        device = x.device

        # 1. 路由计算：完成“输入→专家匹配概率→Top-K 专家选择”
        with profiler.record_function("MoE_Routing"):
            logits = self.gate(
                x
            )  # [batch_size, num_experts]：门控输出各专家的原始匹配度（无范围约束）
            probs = torch.softmax(
                logits, dim=-1
            )  # 将 logits 归一化为 0-1 概率：确保路由权重可解释（概率越高越匹配）
            topk_probs, topk_indices = torch.topk(
                probs, self.top_k, dim=-1
            )  # 取 Top-K 专家：实现稀疏激活，降低计算量

        # 2. 负载均衡损失（仅训练时）：防止专家闲置，确保模型充分利用容量
        if self.training:
            with profiler.record_function("MoE_Auxloss"):
                # [n_experts]
                importance = probs.sum(
                    0
                )  # [num_experts]：每个专家的总路由概率（反映整体重要性）
                importance_loss = torch.var(importance) / (
                    self.num_experts**2
                )  # 归一化方差：避免数值过大

                # 创建 Top-K 掩码：标记哪些专家被选中（用于过滤未选中的专家概率）
                # [N, n_experts]
                mask = torch.zeros_like(probs, dtype=torch.bool)
                mask.scatter_(
                    1, topk_indices, True
                )  # scatter_：按 topk_indices 将 mask 对应位置设为 True
                routing_probs = (
                    probs * mask
                )  # [batch_size, num_experts]：仅保留选中专家的概率
                expert_usage = mask.float().mean(
                    0
                )  # [num_experts]：专家使用率（分配样本占比）
                routing_weights = routing_probs.mean(
                    0
                )  # [num_experts]：专家的平均路由权重（分配样本的依赖度）
                load_balance_loss = (
                    self.num_experts * (expert_usage * routing_weights).sum()
                )  # 归一化损失

                aux_loss = (
                    importance_loss + load_balance_loss
                )  # 总辅助损失：与主任务损失加权求和
        else:
            aux_loss = 0.0  # 推理时无需更新参数，关闭负载均衡损失

        # 3. 专家分配逻辑：建立“样本-选中专家”的映射关系，便于按专家分组计算
        flat_indices = topk_indices.view(
            -1
        )  # [batch_size*top_k]：展平专家索引（如[0,1,2,3]→[0,2,1,3]）
        flat_probs = topk_probs.view(
            -1
        )  # [batch_size*top_k]：展平专家权重（与索引一一对应）

        # 展平样本索引：每个样本对应 top_k 个专家，需标记每个专家索引属于哪个样本
        sample_indices = (
            torch.arange(batch_size, device=device)[:, None]
            .expand(-1, self.top_k)
            .flatten()
        )  # [batch_size*top_k]：如样本 0 对应[0,0]，展平后为[0,0]

        # 4. 专家并行计算：按专家分组处理样本，独立计算后聚合结果
        # 获取输出维度：所有专家输出维度一致，取第一个专家的输出维度即可
        output_dim = self.experts[0].net[-1].out_features
        outputs = torch.zeros(batch_size, output_dim, device=device)  # 初始化输出张量

        with profiler.record_function("MoE_Experts"):
            for expert_idx in range(self.num_experts):
                # 找到分配给当前专家的样本：通过掩码筛选出属于该专家的样本索引
                expert_mask = (
                    flat_indices == expert_idx
                )  # [batch_size*top_k]：True 表示属于当前专家
                expert_samples = sample_indices[expert_mask]  # 属于当前专家的样本 ID
                expert_weights = flat_probs[expert_mask]  # 这些样本对当前专家的权重

                # 容量控制（丢弃超额样本）：避免单个专家处理过多样本导致计算过载或 OOM
                if len(expert_samples) > self.expert_capacity:
                    expert_samples = expert_samples[
                        : self.expert_capacity
                    ]  # 截断至最大容量
                    expert_weights = expert_weights[: self.expert_capacity]

                if len(expert_samples) == 0:
                    continue  # 无样本分配给当前专家，跳过计算

                # 专家计算并加权输出：按公式 y=sum(w_i*E_i(x))，先计算单个专家的加权输出
                expert_output = self.experts[expert_idx](
                    x[expert_samples]
                )  # [num_samples, output_dim]：专家处理样本
                weighted_output = expert_output * expert_weights.unsqueeze(
                    -1
                )  # 权重广播到输出维度（匹配维度后相乘）

                # 聚合结果：将当前专家的加权输出累加到对应样本的位置（一个样本会累加 K 个专家的输出）
                outputs.index_add_(
                    0, expert_samples, weighted_output
                )  # index_add_：按样本 ID 累加，避免循环赋值

        return outputs, aux_loss


class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty([self.n_routed_experts, self.gating_dim])
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bs, seq_len, hidden_size = hidden_states.shape
        # [bs*seq_len, hidden_size]
        hidden_states = hidden_states.view(-1, hidden_size)

        # [bs*seq_len, n_routed_experts]
        logits = F.linear(hidden_states, self.weight)
        # softmax
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        # Top_k select
        # [bs*seq_len, top_k]
        topk_weight, topk_idx = torch.topk(scores, self.top_k, dim=-1)

        # Normalize topk scores if needed
        if self.top_k > 1 and self.norm_topk_prob:
            topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

        # Compute auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            # [bs*seq_len, top_k] -> [bs, seq_len*top_k]
            topk_idx_for_aux_loss = topk_idx.view(bs, -1)
            # 序列级辅助损失
            if self.seq_aux:
                # 计算每个专家的相对负载率
                # 1. 计算每个专家被选中的次数[bs, n_routed_experts]
                # 2. 计算每个专家被选中的理想次数=seq_len * topk / n_routed_experts
                expert_selection_count = torch.zeros(
                    [bs, self.n_routed_experts], device=hidden_states.device
                )

                expert_selection_count.scatter_add_(
                    dim=1,
                    index=topk_idx_for_aux_loss,
                    src=torch.ones_like(topk_idx_for_aux_loss),
                ).div_(seq_len * self.top_k / self.n_routed_experts)
                # 计算每个专家的路由概率平均值[bs, n_routed_experts]
                # [bs*seq_len, n_routed_experts] -> [bs, seq_len, n_routed_experts] -> [bs, n_routed_experts]
                scores_for_aux = scores_for_aux.view(bs, seq_len, -1).mean(dim=1)

                aux_loss = (expert_selection_count * scores_for_aux).sum(
                    1
                ).mean() * self.alpha
            # 批级辅助损失
            else:
                # 计算全局每个专家的相对负载率
                # 1. 计算全局每个专家的平均选择率
                # 2. 乘以n_experts得到每个专家的相对负载因子
                expert_selection_count = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                # [bs*seq_len*top_k, n_routed_experts] -> [n_routed_experts]
                ce = expert_selection_count.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()

        return topk_idx, topk_weight, aux_loss


class MoEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [FeedForward(config) for _ in range(config.n_routed_experts)]
        )

        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [FeedForward(config) for _ in range(config.n_shared_experts)]
            )

    def forward(self, x):
        original = x
        original_shape = x.shape
        bs, seq_len, _ = x.shape
        # [bs*seq_len, topk]
        topk_idx, topk_weight, aux_loss = self.gate(x)
        # [bs*seq_len, hidden_size]
        x = x.view(-1, x.shape[-1])

        # [bs*seq_len*topk]
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            # [bs*seq_len, hidden_size] -> [bs*seq_len*topk, hidden_size]
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x)
            for i, expert in enumerate(self.experts):
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else:
                    y[flat_topk_idx == i] = expert_out + 0 * sum(
                        p.sum() for p in expert.parameters()
                    )
                y = y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1).sum(1)
                y = y.view(*original_shape)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(
                *original_shape
            )
        if self.config.n_shared_experts > 0:
            for expert in range(self.shared_experts):
                y = y + expert(original)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        x: [N, D]
        flat_expert_indices: [N * top_k] [2, 0, 1, 2, 0, 3], 3个专家, topk=2, N=3
        flat_expert_weights: [N * top_k, 1]
        """
        expert_cache = torch.zeros_like(x)

        # [1, 4, 2, 0, 3, 5]
        idxs = flat_expert_indices.argsort()
        # [2, 1, 2, 1] -> [2, 3, 5, 6]
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # [0, 2, 1, 0, 1, 2]
        token_idxs = idxs // self.config.num_experts_per_tok
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue

            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(
                dim=0,
                index=exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
                src=expert_out,
            )

        return expert_cache


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
        self.mlp = FeedForward(args) if not args.use_moe else MoEFeedForward(args)

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
