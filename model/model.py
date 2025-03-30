import math
import inspect
from dataclasses import dataclass
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, num_features, alpha=0.99, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True)
            self.running_mean.mul_(self.alpha).add_(mean * (1 - self.alpha))
            self.running_var.mul_(self.alpha).add_(var * (1 - self.alpha))
            out = (x - mean) / torch.sqrt(var + self.eps)
        else:
            out = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        return out


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.qkv_ln = nn.Linear(config.n_embd, config.n_embd * 3, bias=False)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head

        self.flash = hasattr(F, "scaled_dot_product_attention") and config.flash_attn
        if not self.flash:
            print("WARNING: FlashAttention is not available, using torch instead.")

            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

        def forward(self, x):
            B, T, C = x.shape

            q, k, v = map(
                lambda t: self.qkv_ln(t)
                .view(B, T, self.n_head, 3 * (C // self.n_head))
                .permute(0, 2, 1, 3),
                (x, x, x),
            )

            if self.flash:
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=self.dropout
                )
            else:
                score = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                score = score.masked_fill(self.mask[:, :, T:, T:] == 0, float("-inf"))
                score = F.softmax(score, dim=-1)
                score = self.attn_dropout(score)
                out = score @ v

            out = out.permute(0, 2, 1, 3).contiguous().view(B, T, C)
            out = self.c_proj(out)

            out = self.resid_dropout(out)
            return out


class Router(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.top_k = config.top_k
        self.n_exp = config.n_exp
        assert (
            self.top_k >= 1 and self.top_k <= self.n_exp
        ), "top_k should be between 1 and n_exp"

        self.use_noisy_top_k = config.use_noisy_top_k
        self.train_capacity = config.train_capacity
        self.eval_capacity = config.eval_capacity
        self.min_capacity = config.min_capacity
        self.router_use_full_prec = config.router_use_full_prec

        self.use_aux_loss = config.use_aux_loss
        self.use_router_z_loss = config.use_router_z_loss

        self.w_g = nn.Linear(config.n_embd, self.n_exp, bias=False)
        self.w_noise = (
            nn.Linear(config.n_embd, self.n_exp, bias=False)
            if self.use_noisy_top_k
            else None
        )

    def forward(self, x):
        device_type = x.device.type

        ctx = (
            nullcontext()
            if not self.router_use_full_prec
            else torch.amp.autocast(device_type=device_type, enabled=False)
        )

        with ctx:
            B, T, C = x.shape

            num_tokens = B * T

            logits = self.w_g(x)
            if self.use_noisy_top_k:
                noise = F.softplus(self.w_noise(x))
                noise *= torch.randn_like(logits)
                logits += noise

            if self.use_router_z_loss:
                z_loss = self.compute_router_z_loss(logits)
                MANAGER.add_router_z_loss(z_loss)

            # Find top-k experts
            top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)

            router_probs = torch.full_like(logits, float("-inf"))
            router_probs.scatter_(-1, top_k_indices, top_k_logits)
            router_probs = F.softmax(router_probs, dim=-1)

            if self.use_aux_loss:
                aux_loss = self.compute_aux_loss(router_probs, top_k_indices)
                MANAGER.add_aux_loss(aux_loss)

            # Compute Expert Capacity
            exp_capacity = self.get_capacity(num_tokens)

            exp_mask = F.one_hot(top_k_indices, self.n_exp)
            exp_mask = exp_mask.view(num_tokens, self.top_k, self.n_exp)
            exp_mask = exp_mask.permute(1, 0, 2)  # [top_k, B * T, n_exp]
            exp_rank = exp_mask.reshape(self.top_k * num_tokens, self.n_exp)
            # Calculate the rank of each expert
            exp_rank = torch.cumsum(exp_rank, dim=0) - 1
            exp_rank = exp_rank.reshape(self.top_k, num_tokens, self.n_exp)
            exp_mask *= torch.lt(exp_rank, exp_capacity)
            used_capacity = torch.sum(exp_mask, dim=(0, 1))

            exp_rank = torch.sum(exp_mask * exp_rank, dim=-1)

            router_probs = router_probs.view(num_tokens, self.n_exp)[None, :]
            exp_weights = exp_mask * router_probs

            exp_rank_sc = F.one_hot(exp_rank, self.n_exp)

            cb_weight = torch.sum(
                exp_weights.unsqueeze(3) * exp_rank_sc.unsqueeze(2), dim=0
            )
            sec_mask = cb_weight.bool()

            return used_capacity, cb_weight, sec_mask

    def compute_aux_loss(self, expert_probs: torch.Tensor, indices: torch.Tensor):
        with torch.no_grad():
            one_hot_indices = F.one_hot(
                indices, num_classes=self.n_exp
            )  # [B, T, k, n_exp]
            one_hot_indices = torch.sum(
                one_hot_indices.float(), dim=2
            )  # [B, T, n_exp] (sum over k dimension)
            tokens_per_expert = torch.mean(one_hot_indices.float(), dim=(0, 1))

        # equation (6): compute ratio of router probability allocated to each expert
        prob_per_expert = torch.mean(expert_probs.float(), dim=(0, 1))

        return self.n_exp * torch.sum(prob_per_expert * tokens_per_expert)

    def compute_router_z_loss(self, logits: torch.Tensor):
        z_loss = torch.logsumexp(logits, dim=-1) ** 2.0  # [B, T, n_exp]

        # sum over all tokens and divide by total number of tokens
        return torch.mean(z_loss)

    def get_capacity(self, tokens_per_batch):
        capacity_factor = self.train_capacity if self.training else self.eval_capacity
        capacity = math.floor(
            self.top_k * capacity_factor * tokens_per_batch / self.n_exp
        )
        capacity += capacity % 2  # make sure capacity is an even number
        capacity = max(capacity, self.min_capacity)  # use min capacity
        assert capacity > 0
        return int(capacity)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class MLPExperts(nn.Module):
    """
    implementation of multiple MLP-based experts that can process input
    in batch -- based upon ColossalAI OpenMoE but simple, has optional bias, and
    uses a bmm instead of a loop over a mm for each expert to improve efficiency
    link: https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/moe/experts.py
    """

    def __init__(self, config):
        # TODO: add param init
        super().__init__()
        self.bias = config.bias

        self.c_fc = nn.Parameter(
            torch.empty(config.n_exp, config.n_embd, 4 * config.n_embd)
        )
        self.c_proj = nn.Parameter(
            torch.empty(config.n_exp, 4 * config.n_embd, config.n_embd)
        )
        self.fc_bias = (
            nn.Parameter(torch.empty(config.n_exp, 1, 4 * config.n_embd))
            if self.bias
            else None
        )
        self.proj_bias = (
            nn.Parameter(torch.empty(config.n_exp, 1, config.n_embd))
            if self.bias
            else None
        )
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = torch.bmm(x, self.c_fc)
        if self.bias:
            x += self.fc_bias
        x = self.gelu(x)
        x = torch.bmm(x, self.c_proj)
        if self.bias:
            x += self.proj_bias
        x = self.dropout(x)
        return x


class MOELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = Router(config)  # (noisy) top k router
        self.experts = MLPExperts(config)  # group of MLPs (experts)

    def forward(self, x: torch.Tensor):
        B, T, n_embd = x.size()  # track original shape of input
        num_tokens = B * T

        # pass each token through the router
        used_capacity, exp_weight, exp_mask = self.router(x)

        # flatten out the input
        x = x.view(num_tokens, n_embd)

        # reshape tokens into batches for each expert
        # [n_exp, exp_capacity, B * T] * [B * T, n_embd] -> [n_exp, exp_capacity, n_embd]
        exp_batches = exp_mask.permute(1, 2, 0).type_as(x) @ x

        # compute expert output
        exp_out = self.experts(exp_batches)  # [n_exp, exp_capacity, n_embd]

        # aggregate expert outputs based on router weights
        # eq (2) on page 4 of ST-MoE (https://arxiv.org/abs/2202.08906)
        # similar equations are used for other MoE papers
        exp_weight = exp_weight.view(num_tokens, -1)  # [B * T, n_exp * exp_capacity]
        exp_out = exp_out.view(-1, n_embd)  # [n_exp * exp_capacity, n_embd]
        output = exp_weight @ exp_out  # [B * T, n_embd]

        # resize output before return
        return output.view(B, T, n_embd)


class Block(nn.Module):
    def __init__(self, config, use_moe=True):
        super().__init__()

        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)

        if use_moe:
            self.ffn = MOELayer(config)
        else:
            self.ffn = MLP(config)

    def forward(self, x):
        # apply layer norm to input
        x = self.ln_1(x)

        # apply attention
        x = x + self.attn(x)

        # apply layer norm to attention output
        x = self.ln_2(x)

        # apply feed forward network
        x = x + self.ffn(x)

        return x


@dataclass
class LLMConfig:
    block_size: int = 1024
    vocab_size: int = (
        50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    )
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = (
        True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )

    # MoE-related configs
    n_exp: int = 1  # if n_exp = 1 we just use regular MLP layers
    top_k: int = 2
    use_aux_loss: bool = (
        False  # apply auxiliary loss (from Switch Transformer) in router
    )
    use_router_z_loss: bool = False  # apply router z loss (from ST-MoE)
    use_noisy_top_k: bool = False
    aux_loss_weight: float = (
        0.01  # default setting from Switch Transformer (see top of page 8)
    )
    router_z_loss_weight: float = (
        0.001  # default setting from ST-MoE (see page 8 eq. 6)
    )
    train_capacity: float = 1.25  # default setting from ST-MoE (see top of page 6)
    eval_capacity: float = 2.0
    min_capacity: int = 4  # minimum batch size to send to any single expert
    stride: int = 2  # one in every stride layers are converted to an MoE
    use_switch_tfm_init: bool = False  # use weight init scheme from Switch Transformer
    switch_tfm_init_scale: float = 1.0
    router_use_full_prec: bool = False  # use float32 precision in the router


class LLM(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()

        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        if config.n_exp == 1:
            blocks = nn.ModuleList(
                [Block(config, use_moe=False) for _ in range(config.n_layer)]
            )
        else:
            blocks = []
            for i in range(config.n_layer):
                use_moe = (i % config.stride == 0) and (i != config.n_layer - 1)
                blocks.append(Block(config, use_moe=use_moe))
            blocks = nn.ModuleList(blocks)

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": blocks,
                "ln_f": RMSNorm(config.n_embd),
            }
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight Tieing
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        for n, p in self.named_parameters():
            if n.endswith("c_proj.weight") or n.endswith("experts.c_proj"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    @property
    def num_params(self):
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return n_params

    @torch.no_grad()
    def _init_weights(
        self,
        module,
    ):
        if isinstance(module, nn.Linear):
            if self.config.use_switch_tfm_init:
                scale = self.config.switch_tfm_init_scale

                w_fan_in = module.weight.shape[-1]
                w_std = (scale / w_fan_in) ** 0.5
                torch.nn.init.trunc_normal_(
                    module.weight, mean=0.0, std=w_std, a=-2 * w_std, b=2 * w_std
                )

            else:
                torch.nn.init.normal_(
                    module.weight, mean=0.0, std=self.config.n_embd**-0.5
                )

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, MLPExperts):
            if self.config.use_switch_tfm_init:
                scale = self.config.switch_tfm_init_scale

                c_fc_fan_in = module.c_fc.shape[-2]
                c_fc_std = (scale / c_fc_fan_in) ** 0.5
                torch.nn.init.trunc_normal_(
                    module.c_fc,
                    mean=0.0,
                    std=c_fc_std,
                    a=-2 * c_fc_std,
                    b=2 * c_fc_std,
                )

                c_proj_fan_in = module.c_proj.shape[-2]
                c_proj_std = (scale / c_proj_fan_in) ** 0.5
                torch.nn.init.trunc_normal_(
                    module.c_proj,
                    mean=0.0,
                    std=c_proj_std,
                    a=-2 * c_proj_std,
                    b=2 * c_proj_std,
                )
            else:
                # perform standard (normal) initialization of weights
                torch.nn.init.normal_(module.c_fc, mean=0.0, std=0.02)
                torch.nn.init.normal_(module.c_proj, mean=0.0, std=0.02)

            if module.fc_bias is not None:
                torch.nn.init.zeros_(module.fc_bias)
                torch.nn.init.zeros_(module.proj_bias)

        elif isinstance(module, nn.Embedding):
            # perform standard (normal) initialization of weights
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()

        assert (
            T <= self.config.block_size
        ), "Cannot forward, model block size is exhausted."

        # Add Position
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # embedding
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # final layer norm
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size), targets.view(-1)
            )

            if self.config.n_exp > 1 and self.config.use_aux_loss:
                loss += self.config.aux_loss_weight * MANAGER.aggregate_aux_loss()
                MANAGER.reset_aux_loss()

            if self.config.n_exp > 1 and self.config.use_router_z_loss:
                loss += (
                    self.config.router_z_loss_weight * MANAGER.aggregate_router_z_loss()
                )
                MANAGER.reset_router_z_loss()

        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [
            p
            for n, p in param_dict.items()
            if (p.dim() >= 2 and not n.endswith("bias"))
        ]
        nodecay_params = [
            p for n, p in param_dict.items() if (p.dim() < 2 or n.endswith("bias"))
        ]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"

        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            **extra_args,
        )

        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
