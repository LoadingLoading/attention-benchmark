import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = torch.sqrt(torch.tensor(config.n_embd, dtype=torch.float32))
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        #print(f"Input hidden_states shape: {hidden_states.shape}")
        
        query = self.query(hidden_states)
        #print(f"Query shape: {query.shape}")

        key = self.key(hidden_states)
        #print(f"Key shape: {key.shape}")

        value = self.value(hidden_states)
        #print(f"Value shape: {value.shape}")

        if layer_past is not None:
            past_key, past_value = layer_past
            #print(f"Past key shape: {past_key.shape}, Past value shape: {past_value.shape}")
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)
            #print(f"Updated key shape after concat: {key.shape}")
            #print(f"Updated value shape after concat: {value.shape}")

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / self.scale
        #print(f"Attention scores shape: {attention_scores.shape}")

        if attention_mask is not None:
            #print(f"Attention mask shape before unsqueeze: {attention_mask.shape}")
            #attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.squeeze(1)
            #print(f"Attention mask shape after unsqueeze: {attention_mask.shape}")
            attention_scores = attention_scores + attention_mask
            #print(f"Attention scores shape after adding mask: {attention_scores.shape}")

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        #print(f"Attention probs shape: {attention_probs.shape}")

        context_layer = torch.matmul(attention_probs, value)
        #print(f"Context layer shape: {context_layer.shape}")

        if use_cache:
            present = (key, value)
            #print(f"Present key shape: {key.shape}, Present value shape: {value.shape}")
        else:
            present = None

        outputs = (context_layer,)
        if output_attentions:
            outputs += (attention_probs,)
        
        #print(f"Final output shape: {outputs[0].shape}")
        #if use_cache:
            #print(f"Present shape: {present[0].shape}, {present[1].shape}")

        return outputs if not use_cache else outputs + (present,)
