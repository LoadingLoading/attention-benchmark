#not in use now
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM

'''class LLaMACustomModel(LlamaForCausalLM):
    def __init__(self, config, attention_module_cls):
        super(LLaMACustomModel, self).__init__(config)
        # Replace each attention layer with a custom attention module
        for i, layer in enumerate(self.model.layers):
            layer.self_attn = attention_module_cls(config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        print("use the orignial llama")
        outputs = super(LLaMACustomModel, self).forward(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        if labels is not None:
            # Shift logits and labels for calculating cross-entropy loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits[:shift_labels.size(0)]

            # Calculate loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs[1:]

        return outputs'''

class LlamaCustomAttentionModel(LlamaForCausalLM):
    def __init__(self, config, attention_module_cls):
        super(LlamaCustomAttentionModel, self).__init__(config)
        # Replace each layer's attention module with your custom attention module
        for i, layer in enumerate(self.model.layers):
            layer.self_attn = attention_module_cls(config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Optionally, add any custom behavior here
        outputs = super(LlamaCustomAttentionModel, self).forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
