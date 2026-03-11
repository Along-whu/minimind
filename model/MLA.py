import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KVCache:
    def __init__(self, batch_size, max_length, kv_lora_rank, qk_rope_head_dim, **kwargs):
        self.max_length = max_length
        self.seq_len = 0
        
        # 预分配内存
        self.kv_nope_buffer = torch.zeros((batch_size, max_length, kv_lora_rank), kwargs)
        self.k_ropoe_buffer = torch.zeros((batch_size, max_length, 1, qk_rope_head_dim), kwargs)
        
    def update(self, new_kv_nope, new_k_rope):
        batch_size, new_seq_len, _ = new_k_rope.shape
        
        if self.seq_len + new_seq_len > self.max_length:
            raise ValueError("KV Cache 溢出！")
        
        self.kv_nope_buffer[:, self.seq_len: self.seq_len + new_seq_len, :] = new_kv_nope
        self.k_ropoe_buffer[:, self.seq_len: self.seq_len + new_seq_len, :, :] = new_k_rope
        
        self.seq_len += 1
        
        return (self.kv_nope_buffer[:, :self.seq_len, :], self.k_ropoe_buffer[:, :self.seq_len, :, :])
    
    def reset(self):
        self.seq_len = 0
        

def apply_rope_emb(q, k):
    pass

class MLA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model # 5120
        self.num_heads = config.num_heads # 128
        
        self.q_lora_rank = config.q_lora_rank # 1536
        self.kv_lora_rank = config.kv_lora_rank # 512
        
        self.qk_nope_head_dim = config.qk_nope_head_dim # 128
        self.qk_rope_head_dim = config.qk_rope_head_dim # 64
        self.v_head_dim = config.v_head_dim # 128
        
        self.q_a_proj = nn.Linear(self.d_model, self.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.LayerNorm(self.q_lora_rank)
        
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_nope_head_dim, bias=False)
        self.q_rope_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_rope_head_dim, bias=False)
        
        self.kv_a_proj = nn.Linear(
            self.d_model,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False
        )
        self.kv_a_layernorm = nn.LayerNorm(self.kv_lora_rank)
        
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False
        )
        
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.d_model, bias=False)
        
    def forward(self, hidden_states):
        bs, seq_len, _ = hidden_states.shape
        
        # 1. Query
        q_latent = self.q_a_proj(hidden_states) # [B, L, q_lora_rank]
        q_latent = self.q_a_layernorm(q_latent)
        
        q_nope = self.q_b_proj(q_latent) # [B, L, num_heads * qk_head_dim]
        q_nope = q_nope.view(bs, seq_len, self.num_heads, self.qk_nope_head_dim)
        q_rope = self.q_rope_proj(q_latent) # [B, L, num_heads * qk_rope_dim]
        q_rope = q_rope.view(bs, seq_len, self.num_heads, self.qk_rope_head_dim)
        
        # 2. KV
        kv_mixed = self.kv_a_proj(hidden_states) # [B, L, kv_lora_rank + qk_rope_head_dim]
        kv_latent, k_rope = torch.split(kv_mixed, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_latent = self.kv_a_layernorm(kv_latent)
        
        kv_decompressed = self.kv_b_proj(kv_latent) # [B, L, num_heads * (qk_nope_head_dim + v_head_dim)]
        k_nope, v_nope = torch.split(kv_decompressed, [self.num_heads * self.qk_nope_head_dim, self.num_heads * self.v_head_dim], dim=-1)
        
        k_nope = k_nope.view(bs, seq_len, self.num_heads, self.qk_nope_head_dim)
        v_states = v_nope.view(bs, seq_len, self.num_heads, self.v_head_dim)
        
        k_rope = k_rope.unsqueeze(2) # [B, L, 1, qk_rope_head_dim]
        
        # 3. Rope
        q_rope, k_rope = apply_rope_emb(q_rope, k_rope)
        
        # 4. Attention
        query_states = torch.cat([q_nope, q_rope], dim=-1) # [B, L, num_heads, qk_head_dim + qk_rope_dim]
        key_states = torch.cat([k_nope, k_rope.expand(-1, -1, self.num_heads, -1)], dim=-1) # [B, L, num_heads, qk_nope_dim + qk_rope_dim]
        
        query_states = query_states.transpose(1, 2) # [B, num_heads, L, qk_head_dim + qk_rope_dim]
        key_states = key_states.transpose(1, 2)  # [B, num_heads, L, qk_nope_dim + qk_rope_dim]
        v_states = v_states.transpose(1, 2) # [B, num_heads, L, v_head_dim]
        
        scores = query_states @ key_states.transpose(-2, -1) / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim)
        
        scores = F.softmax(scores, dim=-1)
        
        attn_output = scores @ v_states # [B, num_heads, L, v_head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        attn_output = attn_output.view(bs, seq_len, self.num_heads * self.v_head_dim)
        
        output = self.o_proj(attn_output)
        
        return output
        
        
        
class MLA_Inference(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model # 5120
        self.num_heads = config.num_heads # 128
        
        self.q_lora_rank = config.q_lora_rank # 1536
        self.kv_lora_rank = config.kv_lora_rank # 512
        
        self.qk_nope_head_dim = config.qk_nope_head_dim # 128
        self.qk_rope_head_dim = config.qk_rope_head_dim # 64
        self.v_head_dim = config.v_head_dim # 128
        
        self.q_a_proj = nn.Linear(self.d_model, self.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.LayerNorm(self.q_lora_rank)

        self.q_rope_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_rope_head_dim, bias=False)
        
        self.kv_a_proj = nn.Linear(
            self.d_model,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False
        )
        self.kv_a_layernorm = nn.LayerNorm(self.kv_lora_rank)
        
        # 吸收矩阵
        # q_proj: [num_heads, qk_nope_head_dim, q_lora_rank]
        # k_proj: [num_heads, qk_nope_head_dim, kv_lora_rank]
        # fused_q_k_proj = q_proj.T @ k_proj: [num_heads, q_lora_rank, kv_lora_rank]
        self.register_buffer("fused_q_k_proj", torch.empty((self.num_heads, self.q_lora_rank, self.kv_lora_rank)))
        # v_proj: [num_heads, v_head_dim, kv_lora_rank]
        # o_proj: [d_model, num_heads, v_ead_dim]
        # fused_v_o_proj = (o_proj @ v_proj).T: [num_heads, kv_lora_rank, d_model]
        self.register_buffer("fused_o_v_proj", torch.empty(self.num_heads, self.kv_lora_rank, self.d_model))
        
    def forward(self, hidden_states, kv_cache: KVCache, position_ids=None):
        bsz, seq_len, _ = hidden_states.shape
        
        # 1. 压缩
        q_latent = self.q_a_proj(hidden_states) # [bs, seq_len, q_lora_rank]
        q_latent = self.q_a_layernorm(q_latent)
        
        # q_rope: [bs, seq_len, num_heads, qk_rope_dim]
        q_rope = self.q_rope_proj(q_latent).view(bsz, seq_len, self.num_heads, self.qk_rope_head_dim)
        
        kv_mixed = self.kv_a_proj(hidden_states)
        kv_latent, k_rope = torch.split(kv_mixed, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_latent = self.kv_a_layernorm(kv_latent) 
        k_rope = k_rope.view(bsz, seq_len, 1, self.qk_rope_head_dim)
        
        # 2. 位置编码
        q_rope, k_rope = apply_rope_emb(q_rope, k_rope)
        
        # 3. KV cache
        # new_kv_latent: [bs, new_seq_len, kv_lora_rank]
        # new_k_rope: [bs, new_seq_len, 1, qk_rope_dim]
        new_kv_latent, new_k_rope = kv_cache.update(kv_latent, k_rope)
        new_k_rope = new_k_rope.transpose(1, 2).expand(-1, self.num_heads, -1, -1) # new_k_rope: [bs, num_heads, new_seq_len, qk_rope_dim]
        
        # 4. 矩阵吸收
        # [bs, seq_len, q_lora_rank] --> [bs, seq_len, num_heads * kv_lora_rank]
        q_absorbed_nope = F.linear(q_latent, weight=self.fused_q_k_proj.transpose(1, 2).reshape(self.num_heads * self.kv_lora_rank, -1), bias=False)
        q_absorbed_nope = q_absorbed_nope.view(bsz, seq_len, self.num_heads, self.kv_lora_rank)
        q_absorbed_nope = q_absorbed_nope.transpose(1, 2)
        # or 
        # q_absorbed_nope = torch.enisum('b l q, h q k -> b h l k', q_latent, self.fused_q_k_proj)
        
        # MQA
        scores_nope = q_absorbed_nope @ new_kv_latent.unsqueeze(1).transpose(-2, -1) # [bs, num_heads, seq_len, new_seq_len]
        scores_rope = q_rope.transpose(1, 2) @ new_k_rope.transpose(-2, -1) # [bs, num_heads, seq_len, new_seq_len]
        
        scores = (scores_nope + scores_rope) / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim)
        
        scores = F.softmax(scores, dim=-1)
        
        attn_output = scores @ new_kv_latent.unsqueeze(1) # [bs, num_heads, seq_len, kv_lora_rank]
        attn_output = attn_output.transpose(1, 2).contiguous() # [bs, seq_len, num_heads, kv_lora_rank]
        
        attn_output = F.linear(attn_output.view(bsz, seq_len, -1), weight=self.fused_o_v_proj.permute(2, 0, 1).reshape(self.d_model, -1), bias=False) # [bsz, seq_len, d_model]
        # or 
        # attn_output = torch.einsum('b h l k, h k d -> b l m', attn_output, self.fused_o_v_proj)
        
        return attn_output
        
        
class MLA_vLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model # 5120
        self.num_heads = config.num_heads # 128
        
        self.q_lora_rank = config.q_lora_rank # 1536
        self.kv_lora_rank = config.kv_lora_rank # 512
        
        self.qk_nope_head_dim = config.qk_nope_head_dim # 128
        self.qk_rope_head_dim = config.qk_rope_head_dim # 64
        self.v_head_dim = config.v_head_dim # 128
        
        self.q_a_proj = nn.Linear(self.d_model, self.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.LayerNorm(self.q_lora_rank)
        
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_nope_head_dim, bias=False)
        self.q_rope_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_rope_head_dim, bias=False)
        
        self.kv_a_proj = nn.Linear(
            self.d_model,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False
        )
        self.kv_a_layernorm = nn.LayerNorm(self.kv_lora_rank)
        
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False
        )
        
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.d_model, bias=False)
        
    def forward(self, hidden_states, kv_cache: KVCache, position_ids=None):
        bsz, seq_len, _ = hidden_states.shape
        
        # 1. 压缩Query
        q_latent = self.q_a_layernorm(self.q_a_proj(hidden_states)) # [bs, seq_len, q_lora_rank]
        
        # 2. Query & Key 合并
        q_nope = self.q_b_proj(q_latent) 
        q_nope = q_nope.reshape(bsz, seq_len, self.num_heads, self.qk_nope_head_dim) # [bs, seq_len, h, qk_nope_d]
        q_rope = self.q_rope_proj(q_latent)
        q_rope = q_rope.reshape(bsz, seq_len, self.num_heads, self.qk_rope_head_dim) # [bs, seq_len, h, qk_rope_d]
        
        k_up_proj, v_up_proj = torch.split(self.kv_b_proj.weight,
                                           [self.num_heads * self.qk_nope_head_dim, self.num_heads * self.v_head_dim], dim=0)
        
        # [h, qk_nope, kv_lora_rank]
        k_up_proj = k_up_proj.view(self.num_heads, self.qk_nope_head_dim, self.kv_lora_rank)
        # Q 和 K矩阵吸收
        q_nope_absorbed = torch.einsum('b l h d, h d r -> b l h r', q_nope, k_up_proj) # [bs, seq_len, h, kv_lora_rank]
        
        
        # 3. 压缩KV
        kv_mixed = self.kv_a_proj(hidden_states)
        kv_latent, k_rope = torch.split(kv_mixed, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_latent = self.kv_a_layernorm(kv_latent)
        
        # 4. 对 Q和K apply rope
        q_rope, k_rope = apply_rope_emb(q_rope, k_rope)
        # 拼接 [q_nope; q_rope]
        q_full = torch.cat([q_nope_absorbed, q_rope], dim=-1) # [bs, seq_len, h, kv_lora_rank + q_rope]
        
        # 5. KV cache
        # [bs, kvl, kv_lora_rank] [bs, kvl, qk_rope_dim]
        new_kv_latent, new_k_rope = kv_cache.update(kv_latent, k_rope)
        # 拼接 [k_nope; k_rope]
        k_full = torch.cat([new_kv_latent, new_k_rope], dim=-1) # [bs, kvl, kv_lora_rank + q_rope]
        
        # 6. MQA with kvcache
        
        q_nope_absorbed = q_full.transpose(1, 2) # [bs, h, seq_len, kv_lora_rank + q_rope]
        k_full = k_full.unsqueeze(2).transpose(1, 2) # [bs, 1, kvl, kv_lora_rank + q_rope]
        
        new_kv_latent = new_kv_latent.unsqueeze(2).transpose(1, 2) # [bs, 1, kvl, kv_lora_rank]
        scores = q_nope_absorbed @ k_full.transpose(-2, -1) * 1.0 / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim)# [bs, h, seq_len, kvl]
        scores = F.softmax(scores, dim=-1)
        
        attn_out = scores @ new_kv_latent # [bs, h, seq_len, kv_lora_rank]
        
        # V和O矩阵吸收
        v_up_proj = v_up_proj.view(self.num_heads, self.v_head_dim, self.kv_lora_rank)
        o_proj = self.o_proj.weight.transpose(0, 1) # [h * self.v_head_dim, d_model]
        
        # [bs, h, seq_len, v_head_dim]
        attn_out_v = torch.einsum('b h s d, h v d -> b h s v', attn_out, v_up_proj)
        attn_out_v = attn_out_v.transpose(1, 2).reshape(bsz, seq_len, self.num_heads * self.v_head_dim) # [bs, seq_len, h * v_head_dim]
        
        out = attn_out_v @ o_proj
        
        return out
        
        
        
        