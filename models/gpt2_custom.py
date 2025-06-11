import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from typing import Optional, Tuple, Union

class GPT2CustomAttentionModel(GPT2LMHeadModel):
    def __init__(self, config, attention_module_cls):
        super(GPT2CustomAttentionModel, self).__init__(config)
        
        self.attn_module_class_name = attention_module_cls.__name__
        
        for i, layer in enumerate(self.transformer.h):
            layer.attn = attention_module_cls(config)
        
        if not hasattr(self, 'is_gradient_checkpointing'):
            self.is_gradient_checkpointing = getattr(config, 'gradient_checkpointing', False)

    def _simple_forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None
    ):
        transformer_outputs = super(GPT2CustomAttentionModel, self).forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = transformer_outputs.logits

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_shift_labels = shift_labels.view(-1)
            
            num_labels = flat_shift_labels.size(0)
            flat_shift_logits = flat_shift_logits[:num_labels]

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(flat_shift_logits, flat_shift_labels)

            if hasattr(transformer_outputs, 'past_key_values'):
                 return (loss, logits, transformer_outputs.past_key_values) + transformer_outputs[2:]
            else:
                 return (loss, logits) + transformer_outputs[1:]

        return transformer_outputs

    def _complex_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.transformer.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(past_length, seq_length + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
        
        effective_attention_mask = attention_mask
        if self.attn_module_class_name == "MultiHeadLatentAttention":
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    padding_mask_for_latent = attention_mask[:, None, None, :]
                elif attention_mask.dim() == 4 and attention_mask.shape[1:3] == (1,1):
                    padding_mask_for_latent = attention_mask
                else:
                    raise ValueError(f"For MultiHeadLatentAttention, expected attention_mask of shape [B, S] or [B, 1, 1, S], but got {attention_mask.shape}")
                padding_mask_for_latent = padding_mask_for_latent.to(dtype=self.dtype)
                effective_attention_mask = (1.0 - padding_mask_for_latent) * torch.finfo(self.dtype).min
            else:
                effective_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.transformer.wte(input_ids)
        position_embeds = self.transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.transformer.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.transformer.drop(hidden_states)
        output_shape = (-1, seq_length, hidden_states.size(-1))

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, (block, layer_past) in enumerate(zip(self.transformer.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.is_gradient_checkpointing and self.training:
                if use_cache:
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, use_cache=False, output_attentions=False)
                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    layer_past,
                    effective_attention_mask,
                    head_mask[i] if head_mask is not None else None,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_reentrant=True if hasattr(torch.utils.checkpoint, "use_reentrant") and torch.__version__ < "2.0" else self.config.use_cache 
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=effective_attention_mask,
                    head_mask=head_mask[i] if head_mask is not None else None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                att_idx = 1
                if use_cache:
                    att_idx = att_idx + 1
                if len(outputs) > att_idx:
                    all_self_attentions = all_self_attentions + (outputs[att_idx],)

        hidden_states = self.transformer.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,)
            if presents is not None:
                output += (presents,)
            if all_hidden_states is not None:
                output += (all_hidden_states,)
            if all_self_attentions is not None:
                output += (all_self_attentions,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if self.attn_module_class_name == "MultiHeadLatentAttention":
            return self._complex_forward(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            simple_outputs = self._simple_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            if not (return_dict if return_dict is not None else self.config.use_return_dict):
                 return simple_outputs

            current_loss = None
            current_logits = None
            pkv = None
            
            idx = 0
            if labels is not None:
                current_loss = simple_outputs[idx]
                idx +=1
            current_logits = simple_outputs[idx]
            idx+=1
            
            if labels is not None:
                if len(simple_outputs) > 2: pkv = simple_outputs[2] if use_cache else None
            else:
                if len(simple_outputs) > 1: pkv = simple_outputs[1] if use_cache else None

            return CausalLMOutputWithCrossAttentions(
                loss=current_loss,
                logits=current_logits,
                past_key_values=pkv,
                hidden_states=None,
                attentions=None,
            )