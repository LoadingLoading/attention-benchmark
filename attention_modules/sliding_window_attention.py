import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlidingWindowAttention(nn.Module):
    def __init__(self, config, window_size: int = 3):
        super().__init__()

        self.num_heads = config.n_head
        self.head_dim  = config.n_embd // self.num_heads
        self.scale     = math.sqrt(self.head_dim)            # python float
        self.window_size = window_size

        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key   = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        device = hidden_states.device
        bsz, seq_len, _ = hidden_states.size()

        # ---- project Q K V -------------------------------------------- #
        q = self.query(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.key  (hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        v = self.value(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)                       # (B, H, L, D)
        k = k.transpose(1, 2).transpose(-2, -1)     # (B, H, D, L)

        attn_logits = torch.matmul(q, k) / self.scale   # (B, H, L, L)

        # ---- ▼ NEW attention-mask handling --------------------------- #
        if attention_mask is not None:
            # repo may pass   (B, 1, 1, L)  *or*  (B, 1, 1, L, L)
            if attention_mask.dim() == 5:                  # (B,1,1,L,L)
                attention_mask = attention_mask.squeeze(1) # → (B,1,L,L)

            if attention_mask.dim() == 4:                  # (B,1,L,L) OR (B,1,1,L)
                if attention_mask.size(2) == 1:            # (B,1,1,L) → make square
                    attention_mask = attention_mask.squeeze(2)      # (B,1,L)
                    attention_mask = attention_mask.expand(-1, -1, seq_len) \
                                                 .unsqueeze(2)      # (B,1,L,L)

                # expand heads
                attention_mask = attention_mask.expand(-1, self.num_heads, -1, -1)
            else:
                raise ValueError(
                    f"Unsupported attention_mask shape {attention_mask.shape}"
                )

            attn_logits += attention_mask
        # ---- ▲ attention-mask handling ------------------------------- #

        # ---- sliding-window mask ------------------------------------- #
        arange = torch.arange(seq_len, device=device)
        near   = (arange[:, None] - arange[None, :]).abs() <= self.window_size

        window_mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        window_mask[near] = 0.0
        attn_logits += window_mask.unsqueeze(0).unsqueeze(0)          # (1,1,L,L)

        # ---- soft-max ------------------------------------------------ #
        attn_prob = F.softmax(attn_logits, dim=-1)

        if head_mask is not None:
            attn_prob *= head_mask

        # ---- weighted value sum -------------------------------------- #
        context = torch.matmul(attn_prob, v.transpose(1, 2))          # (B,H,L,D)
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

        outputs = (context,)

        if output_attentions:
            outputs += (attn_prob,)

        if use_cache:
            outputs += ((k, v),)

        return outputs