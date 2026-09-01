#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

#from transformers.configuration_utils import PretrainedConfig

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"

    def __init__(self, **kwargs):
        # Handle nested text_config (newer transformers multimodal config format)
        text_config = kwargs.pop("text_config", None)
        if isinstance(text_config, dict):
            for k, v in text_config.items():
                if k not in kwargs:
                    kwargs[k] = v
        super().__init__(**kwargs)
        # Ensure critical attrs are in __dict__ for direct access.
        # Newer transformers (with heterogeneity integration) may store
        # attributes via internal mechanisms invisible to object.__getattribute__.
        self._sync_attrs(kwargs)

    def _sync_attrs(self, kwargs):
        d = self.__dict__
        # Try to_dict() as backup source (captures attributes stored by parent)
        source = dict(kwargs)
        try:
            td = {k: v for k, v in self.to_dict().items() if not k.startswith('_')}
            for k, v in td.items():
                if k not in source:
                    source[k] = v
        except Exception:
            pass
        for key in ('hidden_size', 'vocab_size', 'intermediate_size',
                    'num_attention_heads', 'num_hidden_layers',
                    'num_key_value_heads', 'pretraining_tp', 'rms_norm_eps',
                    'max_position_embeddings', 'torch_dtype'):
            if key not in d or d[key] is None:
                if key in source:
                    d[key] = source[key]
        if d.get('pretraining_tp') is None:
            d['pretraining_tp'] = 1
        if d.get('rms_norm_eps') is None:
            d['rms_norm_eps'] = 1e-5


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.__dict__.get('pretraining_tp', 1)
        self.vocab_size = config.__dict__.get('vocab_size', config.vocab_size if hasattr(config, 'vocab_size') else 32000)
        self.lm_head = nn.Linear(config.__dict__['hidden_size'], self.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        return_img_hiddens: Optional[bool] = False,
        img_token_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        # Force output_hidden_states=True if we need to extract IMG hiddens
        _output_hidden_states = output_hidden_states
        if return_img_hiddens:
            _output_hidden_states = True

        # Forward to base model; be compatible with HF versions that may pass
        # additional kwargs like `cache_position` during generation
        try:
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=_output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
        except TypeError as e:
            # Drop unknown generation-time kwargs (e.g., cache_position) and retry
            safe_kwargs = dict(kwargs)
            for k in [
                'cache_position',
                'cache_positions',
            ]:
                if k in safe_kwargs:
                    safe_kwargs.pop(k)
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=_output_hidden_states,
                return_dict=return_dict,
                **safe_kwargs,
            )

        # Extract IMG token hidden states if requested
        if return_img_hiddens and labels is not None and img_token_ids is not None:
            from llava.model.img_token_utils import extract_img_token_hiddens
            
            # Get last layer hidden states: (B, L, D)
            hidden_states = outputs.hidden_states[-1]
            
            # Extract IMG token hiddens: (B, K, D)
            img_hiddens = extract_img_token_hiddens(
                hidden_states=hidden_states,
                labels=labels,
                img_token_ids=img_token_ids,
                fill_value=0.0
            )
            
            # Add to outputs
            if return_dict:
                # Can't modify frozen dataclass, so return a dict with img_hiddens
                return {
                    'loss': outputs.loss,
                    'logits': outputs.logits,
                    'past_key_values': outputs.past_key_values,
                    'hidden_states': outputs.hidden_states,
                    'attentions': outputs.attentions,
                    'img_hiddens': img_hiddens,
                }
            else:
                # For tuple output, append at the end
                outputs = outputs + (img_hiddens,)

        return outputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
