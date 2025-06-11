import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearFlexAttention(nn.Module):
    def __init__(self, config):
        super(LinearFlexAttention, self).__init__()
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.scale = torch.sqrt(torch.tensor(config.n_embd, dtype=torch.float32))

    def feature_map(self, x):
        # Simple positive feature map (can be replaced with more complex ones like elu + 1)
        return F.relu(x) + 1

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        # Handle past cache: concatenate past key and value
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)

        # Apply feature map
        query_prime = self.feature_map(query)
        key_prime = self.feature_map(key)

        # Compute key_prime^T * value
        kv = torch.matmul(key_prime.transpose(-1, -2), value)

        # Compute normalizer: key_prime^T * 1
        key_sum = key_prime.sum(dim=-2)  # [batch, d]

        # Compute attention output
        numerator = torch.matmul(query_prime, kv)
        denominator = torch.matmul(query_prime, key_sum.unsqueeze(-1)).clamp(min=1e-6)  # avoid division by zero
        context_layer = numerator / denominator

        present = (key, value) if use_cache else None

        outputs = (context_layer,)
        if output_attentions:
            outputs += (None,)  # Attention probs not explicitly available here
        if use_cache:
            outputs += (present,)

        return outputs

