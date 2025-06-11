import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_func

class FlashAttention(nn.Module):
    def __init__(self, config):
        super(FlashAttention, self).__init__()
        self.query = nn.Linear(config.n_embd, config.n_embd).to(torch.bfloat16)
        self.key = nn.Linear(config.n_embd, config.n_embd).to(torch.bfloat16)
        self.value = nn.Linear(config.n_embd, config.n_embd).to(torch.bfloat16)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.n_embd // self.num_heads

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        #torch.set_anomaly_enabled(True)

        #print(f"Initial hidden_states shape: {hidden_states.shape}")
        hidden_states = hidden_states.to(torch.bfloat16)
        #print(f"Converted hidden_states to bfloat16, shape: {hidden_states.shape}")

        batch_size, seq_length, embed_dim = hidden_states.size()
        #print(f"Batch size: {batch_size}, Sequence length: {seq_length}, Embedding dimension: {embed_dim}")
        query = self.query(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        #query = self.query(hidden_states)
        #print(f"Query shape after Linear: {query.shape}")
        #query = query.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        #print(f"Query shape after view and transpose: {query.shape}")
        key = self.key(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        #key = self.key(hidden_states)
        #print(f"Key shape after Linear: {key.shape}")
        #key = key.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        #print(f"Key shape after view and transpose: {key.shape}")
        value = self.value(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        #value = self.value(hidden_states)
        #print(f"Value shape after Linear: {value.shape}")
        #value = value.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        #print(f"Value shape after view and transpose: {value.shape}")

        attention_output = flash_attn_func(query, key, value, causal=True)
        #print(f"Attention output shape from flash_attn_func: {attention_output.shape}")

        attention_output = attention_output.transpose(1, 2).reshape(batch_size, seq_length, embed_dim)
        #print(f"Attention output shape after transpose and reshape: {attention_output.shape}")

        outputs = (attention_output,)
        if output_attentions:
            outputs += (attention_output,)

        if use_cache:
            present = None
            outputs += (present,)

        return outputs
'''
This is trying for llama, but failed
# flash_attention.py
# attention_modules/flash_attention.py
import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_func
from transformers.models.llama.modeling_llama import DynamicCache  # 确保正确导入

class KeyValueCache:
    def __init__(self, key: torch.Tensor, value: torch.Tensor):
        self.key_cache = [key]
        self.value_cache = [value]

    def to_legacy_cache(self):
        return (torch.cat(self.key_cache, dim=2), torch.cat(self.value_cache, dim=2))

class FlashAttention(nn.Module):
    def __init__(self, config):
        super(FlashAttention, self).__init__()
        self.query = nn.Linear(config.hidden_size, config.hidden_size).to(torch.bfloat16)
        self.key = nn.Linear(config.hidden_size, config.hidden_size).to(torch.bfloat16)
        self.value = nn.Linear(config.hidden_size, config.hidden_size).to(torch.bfloat16)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        print("start")
        # 转换 hidden_states 为 bfloat16
        hidden_states = hidden_states.to(torch.bfloat16)

        batch_size, seq_length, embed_dim = hidden_states.size()


        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)


        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attention_output = flash_attn_func(query, key, value, causal=True)


        attention_output = attention_output.transpose(1, 2).reshape(batch_size, seq_length, embed_dim)

        attn_output = attention_output
        attn_weights = None 

        if use_cache:
            if isinstance(past_key_value, DynamicCache):

                past_key_value.key_cache.append(key)
                past_key_value.value_cache.append(value)
                present_key_value = past_key_value
                #print(f"Updated cache size: {len(past_key_value.key_cache)}")
            else:

                present_key_value = KeyValueCache(key, value)
                #print(f"New cache size: {len(present_key_value.key_cache)}")
        else:
            present_key_value = None
        print("done")


        return attn_output, attn_weights, present_key_value'''
