import os
import torch
import pathlib
import yaml
from dataclasses import dataclass, field
from typing import Optional, List
import transformers
from transformers import AutoConfig

import nltk
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from sensorllm.model import *
from sensorllm.model.chronos_model import *
from sensorllm.train.sensorllm_trainer import SensorLLMTrainer, SensorLLMWeightedCELossTrainer
import logging as logger
from sensorllm.data import make_ts_text_data_module, make_ts_text_data_module_stage2, make_ts_classification_data_module_stage2
import warnings

import evaluate

warnings.filterwarnings("ignore")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    pt_encoder_backbone_ckpt: str = field(default=None)
    tokenize_method: str = field(default='MeanScaleUniformBins',
                                 metadata={"help": "MeanScaleUniformBins or StanNormalizeUniformBins."})
    model_type: Optional[str] = field(default="CasualLM",
                                      metadata={"help": "CasualLM or SequenceClassification."})


@dataclass
class DataArguments:
    dataset: str = field(
        default="usc-had",
        metadata={"help": "usc-had, mhealth, pamap, pamap50, uci, capture24"},
    )
    data_path: str = field(
        default="",
        metadata={"help": "Path to the training data."},
    )
    qa_path: str = field(
        default="",
        metadata={"help": "Path to the training QA data."},
    )
    eval_data_path: str = field(
        default="",
        metadata={"help": "Path to the eval data."},
    )
    eval_qa_path: str = field(
        default="",
        metadata={"help": "Path to the eval QA data."},
    )
    shuffle: bool = field(default=True, metadata={"help": "Whether to shuffle data."})
    ignore_qa_types: List[str] = field(default_factory=lambda: ["trend"])
    preprocess_type: str = field(default='Q',
                                 metadata={"help": "Q or Q+cot."})
    preprocess_type_eval: str = field(default='Q+cot',
                                      metadata={"help": "Q or Q+cot."})
    add_ts_special_token_text: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    num_labels: int = field(
        default=12,
        metadata={
            "help": "Number of output labels."
        },
    )
    use_weighted_loss: bool = field(default=True, metadata={"help": "Use weighted loss for classification model."})

    fix_llm: bool = field(default=True, metadata={"help": "Whether to fix the LLM."})
    fix_ts_encoder: bool = field(default=True, metadata={"help": "Whether to fix the pretrained ts encoder."})
    fix_cls_head: bool = field(default=False, metadata={"help": "Whether to fix the cls head of LLM."})
    stage_2: bool = field(default=False)  # * set True when fine-tuning
    only_stage2: bool = field(default=False)
    # * ts backbone ckpt path
    tune_mm_mlp_adapter: bool = field(default=True)
    metric_for_best_model: str = field(default='eval_loss')


def print_trainable_parameters(model):
    all_param = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            print(f"Layer: {name}, Trainable: {param.requires_grad}")
            trainable_params += param.numel()

    for name, param in model.get_model().pt_encoder_backbone.model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            print(f"Layer: {name}, Trainable: {param.requires_grad}")
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def check_model_parameters(model_or_params, print_details=False):
    device_map = {}

    if hasattr(model_or_params, 'named_parameters'):
        param_iterator = model_or_params.named_parameters()
    elif isinstance(model_or_params, dict):
        param_iterator = model_or_params.items()
    else:
        raise ValueError("Input must be a model or a dictionary of parameters")

    for name, param in param_iterator:
        device = param.device
        shape = tuple(param.shape)
        device_str = str(device)

        if device_str not in device_map:
            device_map[device_str] = []

        device_map[device_str].append((name, shape))

        if print_details:
            print(f"Parameter: {name}")
            print(f"  Shape: {shape}")
            print(f"  Device: {device}")
            print("-" * 50)

    print("\nSummary:")
    for device, params in device_map.items():
        print(f"\nDevice: {device}")
        print(f"Number of parameters: {len(params)}")
        if print_details:
            for name, shape in params:
                print(f"  {name}: {shape}")

    return device_map


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    state_dict = trainer.model.state_dict()
    cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
    if trainer.args.should_save:
        trainer._save(output_dir, state_dict=cpu_state_dict)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.log_level = "info"  # * default is passive(warning)
    # * build logger
    # logger = build_logger(__name__, training_args.output_dir + '/train.log')

    logger.warning(f"Using device: {training_args.device}")

    if not training_args.stage_2:
        # stage 1
        logger.warning("Using model of Stage 1")
        model = SensorLLMStage1LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
    else:
        if model_args.model_type == "CasualLM":
            model = SensorLLMStage2LlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
            )
            logger.warning("Loaded CausalLM model.")
        else:
            logger.warning(f"Loading {data_args.dataset} dataset configs ...")
            with open('./sensorllm/model/ts_backbone.yaml', 'r') as file:
                dataset_configs = yaml.safe_load(file)

            dataset_config = dataset_configs[data_args.dataset]

            id2label = dataset_config["id2label"]
            print(f"Dataset id2label:\n{id2label}")
            label2id = {v: k for k, v in id2label.items()}
            assert training_args.num_labels == len(id2label)
            assert model_args.model_type == "SequenceClassification", f"Undefined model_type {model_args.model_type}"
            model = SensorLLMStage2LlamaForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                num_labels=training_args.num_labels,
                id2label=id2label,
                label2id=label2id,
                cache_dir=training_args.cache_dir,
            )
            logger.warning("Loaded SequenceClassification model.")

    model.config.use_cache = False

    print(f"Default pt_encoder_backbone_ckpt is {model_args.pt_encoder_backbone_ckpt}.")
    model.get_model().load_pt_encoder_backbone_checkpoint(model_args.pt_encoder_backbone_ckpt,
                                                          tc=model_args.tokenize_method)
    pt_backbone_config = AutoConfig.from_pretrained(model_args.pt_encoder_backbone_ckpt)

    assert hasattr(pt_backbone_config, "chronos_config"), "Not a Chronos config file"

    chronos_config = ChronosConfig(**pt_backbone_config.chronos_config)
    chronos_config.tokenizer_class = model_args.tokenize_method
    chronos_tokenizer = chronos_config.create_tokenizer()

    if training_args.fix_llm:
        # * This will fix all the parameters
        model.requires_grad_(False)
        # * fix llama, lm_head
        model.get_model().fix_llm = True
        logger.warning("LLM is fixed. Fix_llm flag is set to True")
        model.get_model().ts_proj.requires_grad_(True)
        model.get_model().pt_encoder_backbone.requires_grad_(True)
    else:
        model.get_model().fix_llm = False
        logger.warning("LLM is trainable. Fix_llm flag is set to False")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if not training_args.fix_ts_encoder:
        # * not fix pretrained ts encoder
        model.get_model().fix_ts_encoder = False
        logger.warning(
            "Pretrained TS backbone is trainable. fix_ts_encoder flag is set to False, ts net grad will be recorded.")
    else:
        model.get_model().fix_ts_encoder = True  # * use with torch.inference_mode to control
        logger.warning(
            "Pretrained TS backbone is fixed. fix_ts_encoder flag is set to True, ts net grad will not be recorded.")

        logger.warning("Set requires_grad of Pretrained TS backbone to False")
        model.get_model().pt_encoder_backbone.requires_grad_(
            False)

    if training_args.tune_mm_mlp_adapter:
        # * not fix the projection layer
        # * may need to set the embed_tokens to require_grad = True if added new tokens
        # * this is done in initialize_tokenizer_ts_backbone_config
        logger.warning("Time-series Projector is trainable. ")
    else:
        model.get_model().ts_proj.requires_grad_(False)
        logger.warning("Time-series Projector is fixed.")

    if model_args.model_type == "SequenceClassification":
        if not training_args.fix_cls_head:
            model.score.requires_grad_(True)
            logger.warning("LLM classification head is trainable. ")
        else:
            model.score.requires_grad_(False)
            logger.warning("LLM classification head is fixed.")

    if not training_args.stage_2:
        # * we assume in stage2, llm, and time-series embedder (and projection layer) can be loaded from the model checkpoint
        model.initialize_tokenizer_ts_backbone_config(tokenizer=tokenizer, device=training_args.device,
                                                      fix_llm=training_args.fix_llm, dataset=data_args.dataset)
    else:
        # * stage2
        if training_args.only_stage2:
            logger.warning("The loaded model haven't been trained in Stage 1. Initializing the LLM tokenizer with new tokens now...")
            model.initialize_tokenizer_ts_backbone_config(tokenizer=tokenizer, device=training_args.device,
                                                          fix_llm=training_args.fix_llm, dataset=data_args.dataset)
        else:
            model.initialize_tokenizer_ts_backbone_config_wo_embedding(tokenizer=tokenizer, dataset=data_args.dataset)

    model.get_model().load_start_end_tokens(dataset=data_args.dataset)

    ts_backbone_config = model.get_model().ts_backbone_config
    data_args.ts_backbone_config = ts_backbone_config

    if not training_args.stage_2:
        logger.warning("Stage 1")
        data_module = make_ts_text_data_module(tokenizer=tokenizer, chronos_tokenizer=chronos_tokenizer,
                                                data_args=data_args)
    else:
        # * stage2
        if model_args.model_type == "CasualLM":
            data_module = make_ts_text_data_module_stage2(tokenizer=tokenizer, chronos_tokenizer=chronos_tokenizer,
                                                          data_args=data_args)
        else:
            assert model_args.model_type == "SequenceClassification", f"Undefined model_type {model_args.model_type} for data_module"
            data_module = make_ts_classification_data_module_stage2(tokenizer=tokenizer,
                                                                    chronos_tokenizer=chronos_tokenizer,
                                                                    label2id=label2id,
                                                                    data_args=data_args)

    if model_args.model_type == "CasualLM":
        trainer = SensorLLMTrainer(model=model,
                                   args=training_args,
                                   tokenizer=tokenizer,
                                   **data_module)
    else:
        assert model_args.model_type == "SequenceClassification", f"Undefined model_type {model_args.model_type} for Trainer"

        metric_f1 = evaluate.load("../metrics/f1")
        metric_acc = evaluate.load("../metrics/accuracy")
        metric_precision = evaluate.load("../metrics/precision")
        metric_recall = evaluate.load("../metrics/recall")

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)

            accuracy = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]

            precision_macro = metric_precision.compute(predictions=predictions, references=labels, average='macro')[
                "precision"]
            recall_macro = metric_recall.compute(predictions=predictions, references=labels, average='macro')["recall"]
            f1_macro = metric_f1.compute(predictions=predictions, references=labels, average='macro')["f1"]

            f1_micro = metric_f1.compute(predictions=predictions, references=labels, average='micro')["f1"]

            precision_per_class = metric_precision.compute(predictions=predictions, references=labels, average=None)[
                "precision"]
            recall_per_class = metric_recall.compute(predictions=predictions, references=labels, average=None)["recall"]
            f1_per_class = metric_f1.compute(predictions=predictions, references=labels, average=None)["f1"]

            results = {
                "f1_macro": f1_macro,
                "f1_micro": f1_micro,
                "accuracy": accuracy,
                "precision_macro": precision_macro,
                "recall_macro": recall_macro
            }

            for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
                results[f"precision_class_{i}"] = p
                results[f"recall_class_{i}"] = r
                results[f"f1_class_{i}"] = f

            return results

        model.config.pad_token_id = tokenizer.pad_token_id
        if training_args.use_weighted_loss:
            logger.warning("Using weighted_loss trainer")
            trainer = SensorLLMWeightedCELossTrainer(model=model,
                                                     args=training_args,
                                                     tokenizer=tokenizer,
                                                     compute_metrics=compute_metrics,
                                                     **data_module)
        else:
            del data_module['class_weights']
            print(data_module.keys)
            trainer = SensorLLMTrainer(model=model,
                                       args=training_args,
                                       tokenizer=tokenizer,
                                       compute_metrics=compute_metrics,
                                       **data_module)

    print_trainable_parameters(model)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)

    # eval_results = trainer.evaluate()
    # print("Evaluation results:", eval_results)


if __name__ == "__main__":
    train()
