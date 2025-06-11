import torch
import torch.nn as nn

class GroupQueryAttention(nn.Module):
    def __init__(self, config, num_groups=8):
        super(GroupQueryAttention, self).__init__()
        self.num_groups = num_groups
        self.num_heads = num_groups  
        self.head_dim = config.n_embd // num_groups
        assert config.n_embd % num_groups == 0,  "n_embd must be divisible by num_groups"

        
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.scale = self.head_dim ** 0.5

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        batch_size, seq_len, hidden_dim = hidden_states.size()



        
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        
        q = q.view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)

        
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale

        
        mask = torch.tril(torch.ones((seq_len, seq_len), device=hidden_states.device)).unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)

        # attention output
        context = torch.matmul(attn_probs, v)  # (batch, num_groups, seq_len, head_dim)

        
        context = context.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)

        outputs = (context,)

        if output_attentions:
            outputs += (attn_probs,)

        if use_cache:
            present = None  
            outputs += (present,)

        return outputs

