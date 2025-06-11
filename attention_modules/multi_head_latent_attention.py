import torch
import torch.nn as nn
from typing import Optional, Tuple, Union # Added for clarity

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config, num_latents=64):
        super(MultiHeadLatentAttention, self).__init__()
        self.num_heads = config.num_attention_heads
        self.embed_dim = config.n_embd

        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
            )

        self.head_dim = self.embed_dim // self.num_heads
        self.num_latents = num_latents
        # Use config to get max sequence length (replace 'n_positions' if needed)
        self.max_seq_len = getattr(config, 'n_positions', 1024) # Default to 1024 if not found
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        self.latents = nn.Parameter(torch.randn(1, num_latents, self.embed_dim))

        self.query = nn.Linear(self.embed_dim, self.embed_dim)
        self.key = nn.Linear(self.embed_dim, self.embed_dim)
        self.value = nn.Linear(self.embed_dim, self.embed_dim)

        # Define projection layer ONCE in init
        self.output_projection = nn.Linear(self.num_latents, self.max_seq_len)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        batch_size, seq_len, _ = hidden_states.size()
        device = hidden_states.device
        dtype = hidden_states.dtype # Propagate dtype from input

        latents = self.latents.expand(batch_size, -1, -1).to(dtype=dtype)

        # Ensure input dtype matches layer dtype for K, V projections
        key_input_dtype = self.key.weight.dtype
        value_input_dtype = self.value.weight.dtype
        query = self.query(latents.to(self.query.weight.dtype)) # Also match query layer dtype
        key = self.key(hidden_states.to(key_input_dtype))
        value = self.value(hidden_states.to(value_input_dtype))


        def split_heads(tensor):
            batch_size_split, seq_len_split, _ = tensor.shape
            new_shape = (batch_size_split, seq_len_split) + (self.num_heads, self.head_dim)
            tensor = tensor.view(new_shape)
            return tensor.permute(0, 2, 1, 3) # [B, H, Seq, D_head]

        query = split_heads(query) # [B, H, L, D_head]
        key = split_heads(key)     # [B, H, S, D_head]
        value = split_heads(value) # [B, H, S, D_head]

        if layer_past is not None:
            past_key, past_value = layer_past
            # Ensure past dtype matches current dtype before cat
            key = torch.cat((past_key.to(key.dtype), key), dim=-2)
            value = torch.cat((past_value.to(value.dtype), value), dim=-2)

        present = (key, value) if use_cache else None
        # Use size(-2) for sequence dimension after permute in split_heads
        key_seq_len = key.size(-2)

        scores = torch.matmul(query, key.transpose(-1, -2)) / self.scale.to(dtype)

        if attention_mask is not None:
             # Assume attention_mask is [B, 1, 1, S_orig] with large negative values
            attention_mask = attention_mask[:, :, :, :key_seq_len] # Slice to match key seq len (handles caching)
            # print(f"Shape of scores before adding mask: {scores.shape}")
            # print(f"Shape of attention_mask before adding: {attention_mask.shape}")
            processed_attention_mask = attention_mask.to(dtype=scores.dtype) # Match dtypes before adding
            scores = scores + processed_attention_mask

        attn_probs = torch.softmax(scores, dim=-1)

        if head_mask is not None:
            attn_probs = attn_probs * head_mask

        # Ensure dtypes match for matmul
        attn_output = torch.matmul(attn_probs.to(value.dtype), value) # [B, H, L, D_head]

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous() # [B, L, H, D_head]
        attn_output = attn_output.view(batch_size, self.num_latents, self.embed_dim) # [B, L, D]


        # Project from latent dimension L back to sequence dimension S
        attn_output_t = attn_output.transpose(1, 2) # [B, D, L]
        # Ensure input dtype matches projection layer dtype
        projected_output = self.output_projection(attn_output_t.to(self.output_projection.weight.dtype)) # [B, D, max_S]

        # Slice the output to match the current input sequence length S
        projected_output_sliced = projected_output[:, :, :seq_len] # [B, D, S]

        # Transpose back to standard sequence format
        final_output = projected_output_sliced.transpose(1, 2).contiguous() # [B, S, D]


        outputs = (final_output,)

        if output_attentions:
            outputs += (attn_probs,)

        if use_cache:
            outputs += (present,)

        return outputs