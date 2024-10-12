import torch
from transformers import StoppingCriteria
import torch.nn as nn
import math
from sensorllm.utils import *
from .chronos_model import *
from transformers import (
    LlamaConfig,
    LlamaModel,
    PreTrainedModel
)
import logging

logger = logging.getLogger(__name__)

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


class BaseSensorLLMModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super(BaseSensorLLMModel, self).__init__(config)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"current dir: {current_dir}")
        ts_config_addr = os.path.join(current_dir, "ts_backbone.yaml")
        self.ts_backbone_config = cfg_from_yaml_file(ts_config_addr)

        # self.ts_backbone_config["ts_backbone_output_dimension"] = self.config.hidden_size

        logger.warning(
            f"The hidden size of LLM is {self.config.hidden_size}."
        )

        backbone_output_dim = self.ts_backbone_config['chronos_model']['encoder_output_dim']
        logger.warning(
            f"{self.ts_backbone_config['chronos_model']['name']} output dim: {self.ts_backbone_config['chronos_model']['encoder_output_dim']}.")
        logger.warning(
            f"Use {self.ts_backbone_config['chronos_model']['projection_hidden_layer']} projection hidden layers.")
        if self.ts_backbone_config['chronos_model']['projection_hidden_layer'] > 0:
            # Add projection layer with linear layers and GELU activation
            projection_layers = []
            last_dim = backbone_output_dim
            for i in range(self.ts_backbone_config['chronos_model']['projection_hidden_layer']):
                projection_layers.append(
                    nn.Linear(last_dim, self.ts_backbone_config['chronos_model']["projection_hidden_dim"][i]))
                projection_layers.append(nn.GELU())
                last_dim = self.ts_backbone_config['chronos_model']["projection_hidden_dim"][i]

            projection_layers.append(nn.Linear(last_dim, self.config.hidden_size))
            self.ts_proj = nn.Sequential(*projection_layers)
            logger.warning(
                f"Each layer with {self.ts_backbone_config['chronos_model']['projection_hidden_dim']} hidden units.")
        else:
            # Single layer
            self.ts_proj = nn.Linear(backbone_output_dim, self.config.hidden_size)
        logger.warning(f"TS projector output dim: {self.config.hidden_size}.")

        self.fix_llm = False
        self.fix_ts_encoder = False

    def load_pt_encoder_backbone_checkpoint(self, checkpoint_path=None, tc=None, torch_dtype=None):
        logger.warning(f"Loading default pt_encoder_backbone_ckpt ...")
        pipeline = ChronosPipeline.from_pretrained(
            self.config.pt_encoder_backbone_ckpt if checkpoint_path is None else checkpoint_path,
            device_map=self.device,
            tc="MeanScaleUniformBins" if tc is None else tc,
            torch_dtype=torch.float32 if torch_dtype is None else torch_dtype,
        )
        self.pt_encoder_backbone = pipeline.model
        self.pt_encoder_backbone.to(self.device)

    def load_start_end_tokens(self, dataset=None):
        logger.warning(f"Loading start_end_tokens dict for {dataset} dataset ...")
        dataset_config = self.ts_backbone_config[dataset]
        self.start_end_tokens = {}
        for key in dataset_config:
            if key.endswith('_start_token_id'):
                end_token_key = key.replace('_start_token_id', '_end_token_id')

                assert end_token_key in dataset_config
                self.start_end_tokens[dataset_config[key]] = dataset_config[end_token_key]

    @torch.no_grad()
    def ts_embed(
            self,
            token_ids: torch.Tensor,
            attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get encoder embeddings for the given time series.

        Parameters
        ----------
        context
            Input series. This is either a 1D tensor, or a list
            of 1D tensors, or a 2D tensor whose first dimension
            is batch. In the latter case, use left-padding with
            ``torch.nan`` to align series of different lengths.

        Returns
        -------
        embeddings, tokenizer_state
            A tuple of two tensors: the encoder embeddings and the tokenizer_state,
            e.g., the scale of the time series in the case of mean scaling.
            The encoder embeddings are shaped (batch_size, context_length, d_model)
            or (batch_size, context_length + 1, d_model), where context_length
            is the size of the context along the time axis if a 2D tensor was provided
            or the length of the longest time series, if a list of 1D tensors was
            provided, and the extra 1 is for EOS.
        """

        embeddings = self.pt_encoder_backbone.encode(
            input_ids=token_ids,
            attention_mask=attention_mask,
        )
        # if str(self.model.device) == 'cuda:1':
        #     print("3", embeddings)

        return embeddings



class BaseSensorLLM(PreTrainedModel):
    def initialize_tokenizer_ts_backbone_config_wo_embedding(self, tokenizer, dataset):
        # * called when stage2 or inference or inference without pre-training, assume tokenizer has time-series tokens
        ts_backbone_config = self.get_model().ts_backbone_config

        default_ts_token = ts_backbone_config["default_ts_token"]  # <ts>
        # print(tokenizer.convert_tokens_to_ids([default_ts_token, "<x_acc_end>", "<y_gyro_start>"]))

        tokenizer.add_tokens([default_ts_token], special_tokens=True)

        # * assert tokenizer has the default_ts_token
        ts_backbone_config["ts_token_id"] = tokenizer.convert_tokens_to_ids([default_ts_token])[0]

        if dataset not in ts_backbone_config:
            raise ValueError(f"Cannot find {dataset} in ts_backbone.yaml file.")

        dataset_config = ts_backbone_config[dataset]

        token_keys = [key for key in dataset_config.keys() if key.startswith('default_') and key.endswith('_token')]
        assert len(token_keys) == dataset_config["channel_num"]*2, f"len(token_keys) ! channel_num*2"
        tokenizer.add_tokens([dataset_config[token_key] for token_key in token_keys], special_tokens=True)

        for token_key in token_keys:
            token_id_key = token_key.replace('default_', '').replace('_token', '_token_id')
            dataset_config[token_id_key] = tokenizer.convert_tokens_to_ids([dataset_config[token_key]])[0]

        special_tokens_dict = dict()
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

        tokenizer.add_special_tokens(special_tokens_dict)

    def initialize_tokenizer_ts_backbone_config(
            self, tokenizer, device, fix_llm=True, dataset='usc-had'
    ):

        ts_backbone_config = self.get_model().ts_backbone_config

        default_ts_token = ts_backbone_config["default_ts_token"]  # <ts>

        tokenizer.add_tokens(
            [default_ts_token], special_tokens=True
        )  # * no need to update embed since it will be replaced

        ts_backbone_config["ts_token_id"] = tokenizer.convert_tokens_to_ids([default_ts_token])[0]

        if dataset not in ts_backbone_config:
            raise ValueError(f"Cannot find {dataset} in ts_backbone.yaml file.")

        dataset_config = ts_backbone_config[dataset]

        token_keys = [key for key in dataset_config.keys() if key.startswith('default_') and key.endswith('_token')]
        assert len(token_keys) == dataset_config["channel_num"]*2, f"len(token_keys) ! channel_num*2"

        num_new_tokens = tokenizer.add_tokens([dataset_config[token_key] for token_key in token_keys], special_tokens=True)

        special_tokens_dict = dict()
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

        num_new_tokens += tokenizer.add_special_tokens(special_tokens_dict)

        self.resize_token_embeddings(
            len(tokenizer)
        )  # ! resize_token_embeddings will make the tokens trainable again

        for token_key in token_keys:
            token_id_key = token_key.replace('default_', '').replace('_token', '_token_id')
            dataset_config[token_id_key] = tokenizer.convert_tokens_to_ids([dataset_config[token_key]])[0]

        if num_new_tokens > 0:
            # Get the input embedding and output embedding of the model
            print("Calculate the average of the input embedding as the initialization value of the new token...")
            input_embeddings = self.get_input_embeddings().weight.data

            # Calculate the average of the input embedding and output embedding as the initialization value of the new token
            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )

            # Initialize the embedding of the new token to the average of the previous tokens
            input_embeddings[-num_new_tokens:] = input_embeddings_avg

            if hasattr(self, 'get_output_embeddings') and callable(getattr(self, 'get_output_embeddings')):
                output_embeddings = self.get_output_embeddings()
                if output_embeddings is not None:
                    print("Calculate the average of the output embedding as the initialization value of the new token...")
                    output_embeddings = self.get_output_embeddings().weight.data
                    output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                        dim=0, keepdim=True
                    )
                    output_embeddings[-num_new_tokens:] = output_embeddings_avg
            else:
                print(
                    f"Output embeddings not available.")

        # Set resize_token_embeddings to a multiple of 8 to improve performance
        T, E = input_embeddings = self.get_input_embeddings().weight.shape
        self.resize_token_embeddings(int(8 * math.ceil(T / 8.0)))

        # need to update the input embedding, but no need to update the output embedding
        for p in self.get_input_embeddings().parameters():
            p.requires_grad = True

        if fix_llm:
            # Save original input embeddings
            self.get_model().orig_embeds_params = [
                self.get_input_embeddings().weight.data.clone().to(device=device)
            ]  # * only tuning the new embeddings

            # Try to fix output embeddings if the method exists
            if hasattr(self, 'get_output_embeddings') and callable(getattr(self, 'get_output_embeddings')):
                output_embeddings = self.get_output_embeddings()
                if output_embeddings is not None:
                    for p in output_embeddings.parameters():
                        p.requires_grad = False
                    print("Setting output embeddings fixed.")
            else:
                print("Output embeddings not available.")
            print(f"Setting {num_new_tokens} new tokens' input embeddings trainable.")
        else:
            self.get_model().orig_embeds_params = None

            # Try to make output embeddings trainable if the method exists
            if hasattr(self, 'get_output_embeddings') and callable(getattr(self, 'get_output_embeddings')):
                output_embeddings = self.get_output_embeddings()
                if output_embeddings is not None:
                    for p in output_embeddings.parameters():
                        p.requires_grad = True
                    print("Setting output embeddings and all input embeddings trainable.")
                else:
                    print("Setting all input embeddings trainable.")
            else:
                print("Output embeddings not available. Setting all input embeddings trainable.")


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = [tokenizer(keyword).input_ids for keyword in keywords]
        self.keyword_ids = [keyword_id[0] for keyword_id in self.keyword_ids if
                            type(keyword_id) is list and len(keyword_id) == 1]
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            for keyword_id in self.keyword_ids:
                if output_ids[0, -1] == keyword_id:
                    return True
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False
