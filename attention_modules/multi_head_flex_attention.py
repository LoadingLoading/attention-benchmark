import torch
import torch.nn as nn

class MultiHeadFlexAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadFlexAttention, self).__init__()
        self.num_heads = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        batch_size, seq_length, _ = hidden_states.size()
        query = self.query(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / self.scale

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value)
        context_layer = context_layer.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)

        if use_cache:
            present = (key, value)
        else:
            present = None

        outputs = (context_layer,)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs if not use_cache else outputs + (present,)
