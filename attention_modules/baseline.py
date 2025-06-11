from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

class BaselineAttention(GPT2Attention):
    def __init__(self, config):
        super().__init__(config)
