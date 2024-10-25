from typing import List, Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
from .utils import *
from contextlib import nullcontext

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
)

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

import logging

logger = logging.getLogger(__name__)


class SensorLLMStage1Config(LlamaConfig):
    model_type = "sensorllmstage1"


class SensorLLMStage1LlamaModel(BaseSensorLLMModel):
    config_class = SensorLLMStage1Config

    def __init__(self, config: LlamaConfig):
        super(SensorLLMStage1LlamaModel, self).__init__(config)

    def forward(
            self,
            input_ids: torch.LongTensor = None,  # B, L
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            ts_token_ids: Optional[List[torch.Tensor]] = None, # B, L_ts
            ts_attention_mask: Optional[List[torch.Tensor]] = None,
            ts_tokenizer_state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # Check the dimensions of input_ids
        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids to be a 2D tensor, but got {input_ids.dim()}D tensor")

        orig_embeds_params = getattr(self, "orig_embeds_params", None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        pt_encoder_backbone = getattr(self, 'pt_encoder_backbone', None)

        if pt_encoder_backbone is not None and (input_ids.shape[1] != 1 or self.training) and ts_token_ids is not None:
            assert type(ts_token_ids) is list

            with torch.no_grad() if self.fix_ts_encoder else nullcontext():
                if self.fix_ts_encoder:
                    self.pt_encoder_backbone.eval()
                ts_features = []
                for ti, am in zip(ts_token_ids, ts_attention_mask):  # * iterate over batch
                    ts_feature = self.ts_embed(ti, am)
                    if torch.any(torch.isnan(ts_feature)) or torch.any(torch.isinf(ts_feature)):
                        raise ValueError("ts_feature has NaN values")
                    ts_features.append(ts_feature[0])
            summed_ts_embeddings = [self.ts_proj(ts_feature) for ts_feature in ts_features]

            new_input_embeds = []
            for cur_input_ids, cur_input_embeds, cur_ts_embeds in zip(
                    input_ids, inputs_embeds, summed_ts_embeddings
            ):  # * input_ids: B, L; input_embeds: B, L, C; summed_ts_embeddings: B, L_ts, C
                cur_ts_embeds = cur_ts_embeds.to(
                    device=cur_input_embeds.device
                )

                num_ts_tokens = cur_ts_embeds.shape[0]  # * number of ts tokens
                total_ts_token_count = (cur_input_ids == self.ts_backbone_config["ts_token_id"]).sum()

                if num_ts_tokens != total_ts_token_count:
                    raise ValueError(
                        f"The window size of time-series tokens ({num_ts_tokens}) and input template ts tokens ({total_ts_token_count}) should be the same.")

                for start_token_id, end_token_id in self.start_end_tokens.items():
                    start_token_count = (cur_input_ids == start_token_id).sum()
                    end_token_count = (cur_input_ids == end_token_id).sum()

                    if start_token_count != end_token_count:
                        raise ValueError(
                            f"The number of {start_token_id} tokens ({start_token_count}) and {end_token_id} tokens ({end_token_count}) should be the same.")

                    start_token_positions = torch.where(cur_input_ids == start_token_id)[0]

                    for start_token_pos in start_token_positions:
                        end_token_pos = start_token_pos + num_ts_tokens + 1
                        total_ts_token_count -= num_ts_tokens

                        if end_token_pos >= len(cur_input_ids) or cur_input_ids[end_token_pos] != end_token_id:
                            raise ValueError(
                                f"The end token '{end_token_id}' should follow the start token '{start_token_id}' after {num_ts_tokens} positions."
                            )

                        if orig_embeds_params is not None:  # * will not update the original embeddings except for TS_START_TOKEN and TS_END_TOKEN
                            # print("Will not update the original embeddings except for TS_START_TOKEN and TS_END_TOKEN")
                            cur_input_embeds = torch.cat(
                                (
                                    cur_input_embeds[:start_token_pos].detach(),
                                    cur_input_embeds[start_token_pos: start_token_pos + 1],
                                    cur_ts_embeds,
                                    cur_input_embeds[end_token_pos: end_token_pos + 1],
                                    cur_input_embeds[end_token_pos + 1:].detach(),
                                ),
                                dim=0,
                            )
                        else:
                            # print("Will update the original embeddings")
                            cur_input_embeds = torch.cat(
                                (
                                    cur_input_embeds[:start_token_pos + 1],
                                    cur_ts_embeds,
                                    cur_input_embeds[end_token_pos:],
                                ),
                                dim=0,
                            )

                if total_ts_token_count != 0:
                    raise ValueError(
                        f"The value of total_ts_token_count ({total_ts_token_count}) should be the 0.")
                new_input_embeds.append(cur_input_embeds)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(SensorLLMStage1LlamaModel, self).forward(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )


class SensorLLMStage1LlamaForCausalLM(BaseSensorLLM, LlamaForCausalLM):
    config_class = SensorLLMStage1Config

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = SensorLLMStage1LlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,  # * control whether to return past_key_values
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            ts_token_ids: Optional[List[torch.Tensor]] = None,  # B, L_ts
            ts_attention_mask: Optional[List[torch.Tensor]] = None,
            ts_tokenizer_state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            ts_token_ids=ts_token_ids,
            ts_attention_mask=ts_attention_mask,
            ts_tokenizer_state=ts_tokenizer_state,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()  # * B, L, V
            shift_labels = labels[..., 1:].contiguous()  # * B, L
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "ts_token_ids": kwargs.get("ts_token_ids", None),
                "ts_attention_mask": kwargs.get("ts_attention_mask", None),
                "ts_tokenizer_state": kwargs.get("ts_tokenizer_state", None),
                "cache_position": kwargs.get("cache_position", None),
            }
        )
        return model_inputs


AutoConfig.register("sensorllmstage1", SensorLLMStage1Config)
AutoModelForCausalLM.register(SensorLLMStage1Config, SensorLLMStage1LlamaForCausalLM)
