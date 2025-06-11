#not in use now

from transformers import LlamaForCausalLM
from attention_modules.flash_attention import FlashAttention

class SmolLM2CustomAttentionModel(LlamaForCausalLM):
    def __init__(self, config, attention_module_cls):
        super(SmolLM2CustomAttentionModel, self).__init__(config)
        for i, layer in enumerate(self.model.layers):
            layer.self_attn = attention_module_cls(config)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = super(SmolLM2CustomAttentionModel, self).forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=True
        )
        return outputs
