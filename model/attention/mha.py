import torch
import torch.nn as nn

import sys
import os
import math

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__), "../..")))
from config.model_config import ModelConfig


class MultiHeadedAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        assert (
            config.embed_dim % config.attention.num_heads == 0
        ), "embed_dim must be divisible by num_heads"

        self.num_heads = config.attention.num_heads
        self.head_dim = config.embed_dim // config.attention.num_heads

        self.qkv_ln = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=False)
        self.out_ln = nn.Linear(config.embed_dim, config.embed_dim, bias=False)

        # Regularization
        self.attn_dropout = nn.Dropout(config.attention.dropout_rate)

        # Whether support flash attention
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        if not self.flash:
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(self.head_dim, self.head_dim)).view(
                    1, 1, self.head_dim, self.head_dim
                ),
            )

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = list(
            map(
                lambda f: f(x).view(B, T, self.num_heads, self.head_dim),
                self.qkv_ln(x).chunk(3, dim=-1),
            )
        )

        if self.flash:
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0,
                is_causal=True,
            )
        else:
            out = torch.einsum("bthd,bthd->bth", q, k) / math.sqrt(self.head_dim)
            out = out.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf")).softmax(
                dim=-1
            )
            out = self.attn_dropout(out)
            out = torch.einsum("bthd,bth->bthd", v, out)

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_ln(out)
