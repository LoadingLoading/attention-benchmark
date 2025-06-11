import torch
import torch.nn as nn

class SparseFlexAttention(nn.Module):
    def __init__(self, config, sparsity=0.1):
        super(SparseFlexAttention, self).__init__()
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.sparsity = sparsity
        self.scale = torch.sqrt(torch.tensor(config.n_embd, dtype=torch.float32))

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / self.scale
        mask = torch.rand(attention_scores.size(), device=hidden_states.device) < self.sparsity
        attention_scores = attention_scores.masked_fill(mask, float('-inf'))

        if attention_mask is not None:
            attention_mask = attention_mask.to(hidden_states.device)
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value)

        if use_cache:
            present = (key, value)
        else:
            present = None

        outputs = (context_layer,)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs if not use_cache else outputs + (present,)
