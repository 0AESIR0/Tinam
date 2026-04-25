"""
model.py
Sıfırdan yazılmış küçük GPT-2 tarzı decoder-only Transformer.
1M–5M parametre arası konfigürasyonlar aşağıda tanımlı.
"""

import math
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TINAConfig(PretrainedConfig):
    model_type = "tina"

    def __init__(
        self,
        vocab_size: int     = 8000,
        n_embd: int         = 256,
        n_layer: int        = 6,
        n_head: int         = 8,
        n_ctx: int          = 512,       # maksimum sequence uzunluğu
        dropout: float      = 0.1,
        mlp_ratio: float    = 4.0,       # FFN gizli boyutu = n_embd * mlp_ratio
        bias: bool          = False,     # QKV / Linear bias
        tie_weights: bool   = True,      # embedding ↔ lm_head ağırlık paylaşımı
        pad_token_id: int   = 1,
        bos_token_id: int   = 2,
        eos_token_id: int   = 3,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.n_embd     = n_embd
        self.n_layer    = n_layer
        self.n_head     = n_head
        self.n_ctx      = n_ctx
        self.dropout    = dropout
        self.mlp_ratio  = mlp_ratio
        self.bias       = bias
        self.tie_weights = tie_weights


# ---------------------------------------------------------------------------
# Hazır konfigürasyonlar  (param sayısı yaklaşık)
# ---------------------------------------------------------------------------

PRESETS = {
    # ~1.2M param  (hızlı deneme, az veriyle bile öğrenir)
    "micro": dict(n_embd=128, n_layer=4, n_head=4, n_ctx=256),

    # ~3M param  (önerilen başlangıç noktası)
    "small": dict(n_embd=256, n_layer=6, n_head=8, n_ctx=512),

    # ~5M param  (daha fazla veri ve GPU zamanı gerektirir)
    "medium": dict(n_embd=384, n_layer=8, n_head=8, n_ctx=512),
}


# ---------------------------------------------------------------------------
# Bloklar
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (bias yok, daha verimli)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.scale * x * rms


class RotaryEmbedding(nn.Module):
    """RoPE — Rotary Position Embedding."""
    def __init__(self, dim: int, max_seq: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq)

    def _build_cache(self, seq_len: int):
        t     = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q, k, cos, sin):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TINAConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head  = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.scale   = self.head_dim ** -0.5

        self.qkv  = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd,     bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)
        self.rope  = RotaryEmbedding(self.head_dim, max_seq=cfg.n_ctx)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[tuple] = None,
        use_cache: bool = False,
    ):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)

        # [B, heads, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # RoPE uygula
        cos, sin = self.rope(q, T)
        q, k = apply_rotary(q, k, cos, sin)

        # KV cache
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present = (k, v) if use_cache else None

        # Flash attention (PyTorch ≥2.0)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.drop.p if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), present


class MLP(nn.Module):
    def __init__(self, cfg: TINAConfig):
        super().__init__()
        hidden = int(cfg.n_embd * cfg.mlp_ratio)
        self.fc1  = nn.Linear(cfg.n_embd, hidden,      bias=cfg.bias)
        self.fc2  = nn.Linear(hidden,     cfg.n_embd,  bias=cfg.bias)
        self.gate = nn.Linear(cfg.n_embd, hidden,      bias=cfg.bias)  # SwiGLU gate
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # SwiGLU aktivasyonu
        gate = torch.nn.functional.silu(self.gate(x))
        return self.drop(self.fc2(gate * self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TINAConfig):
        super().__init__()
        self.ln1  = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = RMSNorm(cfg.n_embd)
        self.mlp  = MLP(cfg)

    def forward(self, x, past_kv=None, use_cache=False):
        attn_out, present = self.attn(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, present


# ---------------------------------------------------------------------------
# Ana model
# ---------------------------------------------------------------------------

class TINAModel(PreTrainedModel):
    config_class = TINAConfig

    def __init__(self, cfg: TINAConfig):
        super().__init__(cfg)
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd, padding_idx=cfg.pad_token_id)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f    = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # GPT-2 style scaled init for residual projections
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        B, T = input_ids.shape

        x = self.drop(self.tok_emb(input_ids))

        past_key_values = past_key_values or [None] * self.cfg.n_layer
        presents = []

        for i, block in enumerate(self.blocks):
            x, present = block(x, past_kv=past_key_values[i], use_cache=use_cache)
            presents.append(present)

        x      = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # shift
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss(ignore_index=self.cfg.pad_token_id)(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=presents if use_cache else None,
        )

    @torch.no_grad()
    def generate_text(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int = 3,
    ) -> torch.Tensor:
        """Basit top-k + top-p (nucleus) sampling."""
        self.eval()
        past = None
        for _ in range(max_new_tokens):
            out = self(input_ids if past is None else input_ids[:, -1:],
                       past_key_values=past, use_cache=True)
            past   = out.past_key_values
            logits = out.logits[:, -1, :] / max(temperature, 1e-6)

            # top-k
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # top-p (nucleus)
            probs = torch.softmax(logits, dim=-1)
            sorted_p, sorted_idx = torch.sort(probs, descending=True)
            cum_p = sorted_p.cumsum(-1)
            remove = cum_p - sorted_p > top_p
            sorted_p[remove] = 0.0
            sorted_p /= sorted_p.sum(-1, keepdim=True)
            next_token = sorted_idx.gather(-1, torch.multinomial(sorted_p, 1))

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if (next_token == eos_token_id).all():
                break

        return input_ids
