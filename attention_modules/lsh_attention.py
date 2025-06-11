import torch
import torch.nn as nn

class LSHAttention(nn.Module):
    def __init__(self, config, num_hashes=8):
        super(LSHAttention, self).__init__()
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.num_hashes = num_hashes
        self.scale = torch.sqrt(torch.tensor(config.n_embd, dtype=torch.float32))

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        hash_buckets = torch.randint(0, self.num_hashes, (query.size(0), query.size(1)), device=query.device)
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / self.scale
        attention_scores[hash_buckets.unsqueeze(-1) != hash_buckets.unsqueeze(-2)] = float('-inf')

        if attention_mask is not None:
            #attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.squeeze(1)
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
