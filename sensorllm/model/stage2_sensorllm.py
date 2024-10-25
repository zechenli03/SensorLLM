from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from .utils import *
from contextlib import nullcontext
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaForSequenceClassification
)

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast
)

import logging

logger = logging.getLogger(__name__)


class SensorLLMStage2Config(LlamaConfig):
    model_type = "sensorllmstage2"


class SensorLLMStage2LlamaModel(BaseSensorLLMModel):
    config_class = SensorLLMStage2Config

    def __init__(self, config: LlamaConfig):
        super(SensorLLMStage2LlamaModel, self).__init__(config)

    def forward(
            self,
            input_ids: torch.LongTensor = None,  # B, L
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            mts_token_ids: Optional[torch.Tensor] = None,
            mts_attention_mask: Optional[torch.Tensor] = None,
            mts_tokenizer_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # Check the dimensions of input_ids
        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids to be a 2D tensor, but got {input_ids.dim()}D tensor")

        if mts_token_ids.dim() != 3:
            raise ValueError(f"Expected multichannel_ts to be a 3D tensor, but got {mts_token_ids.dim()}D tensor")

        orig_embeds_params = getattr(self, "orig_embeds_params", None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        pt_encoder_backbone = getattr(self, 'pt_encoder_backbone', None)

        if pt_encoder_backbone is not None and (input_ids.shape[1] != 1 or self.training) and mts_token_ids is not None:

            channel_num = mts_token_ids.size(1)
            with torch.no_grad() if self.fix_ts_encoder else nullcontext():
                if self.fix_ts_encoder:
                    self.pt_encoder_backbone.eval()
                ts_features = []
                for ts_token_ids, ts_attention_mask in zip(mts_token_ids, mts_attention_mask):  # * iterate over batch
                    ts_feature = self.ts_embed(ts_token_ids, ts_attention_mask)
                    if torch.any(torch.isnan(ts_feature)) or torch.any(torch.isinf(ts_feature)):
                        raise ValueError("ts_feature has NaN values")
                    ts_features.append(ts_feature)

            summed_ts_embeddings = [self.ts_proj(ts_feature) for ts_feature in ts_features]
            summed_ts_embeddings = torch.stack(summed_ts_embeddings)

            new_input_embeds = []
            for cur_input_ids, cur_input_embeds, cur_ts_embeds in zip(
                    input_ids, inputs_embeds, summed_ts_embeddings
            ):  # * input_ids: B, L; input_embeds: B, L, D; summed_ts_embeddings: B, C, L_ts, D
                cur_ts_embeds = cur_ts_embeds.to(
                    device=cur_input_embeds.device
                )
                num_ts_tokens = cur_ts_embeds.shape[1]  # * number of ts tokens

                if len(self.start_end_tokens) != channel_num:
                    raise ValueError(
                        f"The length of start_end_tokens ({len(self.start_end_tokens)}) and channel_num ({channel_num}) should be the same.")
                if len(self.start_end_tokens) != cur_ts_embeds.size(0):
                    raise ValueError(
                        f"The length of start_end_tokens ({len(self.start_end_tokens)}) and cur_ts_embeds ({cur_ts_embeds.size(0)}) should be the same.")

                total_ts_token_count = (cur_input_ids == self.ts_backbone_config["ts_token_id"]).sum()
                for (start_token_id, end_token_id), channel_ebd in zip(self.start_end_tokens.items(), cur_ts_embeds):
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
                            cur_input_embeds = torch.cat(
                                (
                                    cur_input_embeds[:start_token_pos].detach(),
                                    cur_input_embeds[start_token_pos: start_token_pos + 1],
                                    channel_ebd,
                                    cur_input_embeds[end_token_pos: end_token_pos + 1],
                                    cur_input_embeds[end_token_pos + 1:].detach(),
                                ),
                                dim=0,
                            )
                        else:
                            cur_input_embeds = torch.cat(
                                (
                                    cur_input_embeds[:start_token_pos + 1],
                                    channel_ebd,
                                    cur_input_embeds[end_token_pos:],
                                ),
                                dim=0,
                            )
                if total_ts_token_count != 0:
                    raise ValueError(
                        f"The value of total_ts_token_count ({total_ts_token_count}) should be the 0.")
                new_input_embeds.append(cur_input_embeds)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(SensorLLMStage2LlamaModel, self).forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class SensorLLMStage2LlamaForCausalLM(BaseSensorLLM, LlamaForCausalLM):
    config_class = SensorLLMStage2Config

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = SensorLLMStage2LlamaModel(config)

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
            mts_token_ids: Optional[torch.Tensor] = None,
            mts_attention_mask: Optional[torch.Tensor] = None,
            mts_tokenizer_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            return_dict: Optional[bool] = None,
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
            mts_token_ids=mts_token_ids,
            mts_attention_mask=mts_attention_mask,
            mts_tokenizer_state=mts_tokenizer_state
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
                "mts_token_ids": kwargs.get("mts_token_ids", None),
                "mts_attention_mask": kwargs.get("mts_attention_mask", None),
                "mts_tokenizer_state": kwargs.get("mts_tokenizer_state", None),
            }
        )
        model_inputs.pop("cache_position")
        return model_inputs


class SensorLLMStage2LlamaForSequenceClassification(BaseSensorLLM, LlamaForSequenceClassification):
    config_class = SensorLLMStage2Config

    def __init__(self, config):
        super(LlamaForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.model = SensorLLMStage2LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

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
            use_cache: Optional[bool] = None, # * control whether to return past_key_values
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            mts_token_ids: Optional[torch.Tensor] = None,
            mts_attention_mask: Optional[torch.Tensor] = None,
            mts_tokenizer_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
                labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                    Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                    config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                    `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
                """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mts_token_ids=mts_token_ids,
            mts_attention_mask=mts_attention_mask,
            mts_tokenizer_state=mts_tokenizer_state
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


AutoConfig.register("sensorllmstage2", SensorLLMStage2Config)
AutoModelForCausalLM.register(SensorLLMStage2Config, SensorLLMStage2LlamaForCausalLM)
AutoModelForSequenceClassification.register(SensorLLMStage2Config, SensorLLMStage2LlamaForSequenceClassification)