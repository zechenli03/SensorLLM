from typing import Dict, Sequence
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random
import copy
import json
import re
import pickle
import logging
from dataclasses import dataclass

from sensorllm.data.utils import generate_chat_template, preprocess, preprocess_cls, get_token_list
from sensorllm.model.chronos_model import *
import transformers

import torch

IGNORE_INDEX = -100
RDM_SEED = 42


def preprocess_time_series_stage2(
        sources: Sequence[Dict[str, str]], # [{"Q": "...", "A": "..."}]
        added_str: str,
) -> Sequence[Dict[str, str]]:
    modified_sources = []
    pattern = r'\b\d+\.\s+[A-Za-z\s]+\.'
    for index, source in enumerate(sources):
        modified_q = added_str + source["Q"]

        matches = re.findall(pattern, source["A"])
        assert len(matches) == 1
        cot = source["A"].replace(matches[-1], '')

        modified_sources.append({"Q": modified_q, "A": source["A"], "cot": cot.strip(), "ground_truth": matches[-1]})
    return modified_sources


def preprocess_time_series_CLS_stage2(
        sources: Sequence[Dict[str, str]], # [{"Q": "...", "A": "..."}]
) -> Sequence[Dict[str, str]]:
    modified_sources = []
    for index, source in enumerate(sources):
        modified_sources.append({
            "Q": source.get("Q", ""),
            "smry": source.get("smry", ""),
            "trend_text": source.get("trend_text", ""),
            "corr_text": source.get("corr_text", ""),
            "info_text": source.get("info_text", ""),
            "answer": source.get("A", ""),
            "label": source.get("label", "")
        })
    return modified_sources


class MultiChannelTimeSeriesDatasetStage2(Dataset):
    def __init__(self, data_path=None, qa_path=None, tokenizer=None, chronos_tokenizer=None, split=None, data_args=None):
        super(MultiChannelTimeSeriesDatasetStage2, self).__init__()
        self.data_path = data_path
        self.qa_path = qa_path
        self.tokenizer = tokenizer
        self.chronos_tokenizer = chronos_tokenizer
        self.split = split
        self.preprocess_type = data_args.preprocess_type
        self.preprocess_type_eval = None if tokenizer is None else data_args.preprocess_type_eval

        shuffle = data_args.shuffle
        dataset = data_args.dataset

        if dataset == 'usc-had':
            self.SYS_INST = "The assistant is provided with time-series readings of six sensor channels, including three accelerometer channels (in g) and three gyroscope channels (in dps). Each channel contains 200 data representing information extracted from the same 2-second time window at a sampling rate of 100Hz. Please analyze the trends and patterns in each channel to identify the correct activity type from the following twelve activity options:\n\n1. Walking Forward\n2. Walking Left\n3. Walking Right\n4. Walking Upstairs\n5. Walking Downstairs\n6. Running Forward\n7. Jumping\n8. Sitting\n9. Standing\n10. Sleeping\n11. Elevator Up\n12. Elevator Down\n\nProvide the predicted activity as both the number and the name at the end."
        elif dataset == 'capture24':
            self.SYS_INST = "The assistant is provided with time-series readings of three accelerometer channels (in g). Each channel contains 500 data representing information extracted from the same 10-second time window at a sampling rate of 50Hz. Please analyze the trends and patterns in each channel to identify the correct activity type from the following 10 activity options:\n\n1. sleep\n2. sitting\n3. household-chores\n4. walking\n5. vehicle\n6. bicycling\n7. mixed-activity\n8. standing\n9. manual-work\n10. sports\n\nProvide the predicted activity as both the number and the name at the end."
        elif dataset == 'mhealth':
            self.SYS_INST = "The assistant is provided with time-series readings of 15 sensor channels, including acceleration sensors (in m/s^2) and gyroscope sensors (in deg/s). Each channel contains 100 data representing information extracted from the same 2-second time window at a sampling rate of 50Hz. Please analyze the trends and patterns in each channel to identify the correct activity type from the following twelve activity options:\n\n1. Standing still\n2. Sitting and relaxing\n3. Lying down\n4. Walking\n5. Climbing stairs\n6. Waist bends forward\n7. Frontal elevation of arms\n8. Knees bending (crouching)\n9. Cycling\n10. Jogging\n11. Running\n12. Jump front & back\n\nProvide the predicted activity as both the number and the name at the end."
        elif dataset == 'pamap50':
            self.SYS_INST = "The assistant is provided with time-series readings of 27 sensor channels, including acceleration sensors (in m/s^2) and gyroscope sensors (in rad/s) and magnetometer sensors (in μT). Each channel contains 100 data representing information extracted from the same 2-second time window at a sampling rate of 50Hz. Please analyze the trends and patterns in each channel to identify the correct activity type from the following twelve activity options:\n\n1. lying\n2. sitting\n3. standing\n4. walking\n5. running\n6. cycling\n7. Nordic walking\n8. ascending stairs\n9. descending stairs\n10. vacuum cleaning\n11. ironing\n12. rope jumping\n\nProvide the predicted activity as both the number and the name at the end."
        elif dataset == 'pamap':
            self.SYS_INST = "The assistant is provided with time-series readings of 27 sensor channels, including acceleration sensors (in m/s^2) and gyroscope sensors (in rad/s) and magnetometer sensors (in μT). Each channel contains 100 data representing information extracted from the same 2-second time window at a sampling rate of 100Hz. Please analyze the trends and patterns in each channel to identify the correct activity type from the following twelve activity options:\n\n1. lying\n2. sitting\n3. standing\n4. walking\n5. running\n6. cycling\n7. Nordic walking\n8. ascending stairs\n9. descending stairs\n10. vacuum cleaning\n11. ironing\n12. rope jumping\n\nProvide the predicted activity as both the number and the name at the end."
        elif dataset == 'uci':
            self.SYS_INST = "The assistant is provided with time-series readings of six sensor channels, including three accelerometer channels (in g) and three gyroscope channels (in dps). Each channel contains 128 data representing information extracted from the same 2.56-second time window at a sampling rate of 50Hz. Please analyze the trends and patterns in each channel to identify the correct activity type from the following twelve activity options:\n\n1. Walking Forward\n2. Walking Upstairs\n3. Walking Downstairs\n4. Sitting\n5. Standing\n6. Laying\n\nProvide the predicted activity as both the number and the name at the end."
        else:
            raise ValueError(f"Wrong dataset name in __init__: {dataset}")

        self.ts_data, self.list_data_dict = self._flatten_data(shuffle)

        self.data_args = data_args.ts_backbone_config

        self.window_length = len(self.ts_data[0][0])
        self.channel_num = len(self.ts_data[0])
        assert self.channel_num == self.data_args[dataset]["channel_num"], "channel_num, data_args.channel_num shape mismatched"

        print(
            f"The dataset size is: {len(self.ts_data)}. Window size: {self.window_length}. Channel num: {self.channel_num}."
        )

        if self.data_args["chronos_model"]["last_token"]:
            added_token = self.data_args["default_ts_token"] * (self.window_length + 1)
        else:
            added_token = self.data_args["default_ts_token"] * self.window_length

        start_tokens_list, end_tokens_list = get_token_list(dataset, self.data_args[dataset], data_args.add_ts_special_token_text)

        added_str = ''
        for start_token, end_token in zip(start_tokens_list, end_tokens_list):
            added_str += start_token + added_token + end_token + '\n'
        self.added_str = added_str

    def _flatten_data(self, shuffle: bool):
        logging.warning(f"Loading {self.split} data...")
        with open(self.data_path, "rb") as f:
            data_file = pickle.load(f)
        with open(self.qa_path, "r") as file:
            qa_file = json.load(file)
        data_file = np.array(data_file, dtype=np.float64)
        ts_data = []
        qa_dict = []
        assert len(data_file) == len(qa_file["dataset"])
        for q in qa_file["dataset"]:
            data_idx = q["index"]
            data = data_file[int(data_idx)]
            ts_data.append([torch.from_numpy(data[:, i]).to(torch.float64) for i in range(data.shape[1])])
            qa_dict.append(q["qa_pair"])
        assert len(ts_data) == len(qa_dict), "ts_data, qa_dict, length not matched"

        if shuffle:
            print("Shuffling data...")
            random.seed(RDM_SEED)
            indexes = list(range(len(ts_data)))
            random.shuffle(indexes)
            ts_data = [ts_data[i] for i in indexes]
            qa_dict = [qa_dict[i] for i in indexes]
        #
        # if self.split == "eval":
        #     return ts_data[:100], qa_dict[:100]

        return ts_data, qa_dict

    def __len__(self):
        return len(self.ts_data)

    def __getitem__(self, index):
        sources = self.list_data_dict[index]  # {"Q": ..., "A": ...}
        multichannel_ts = self.ts_data[index]  # C * L, 6 * 200

        if isinstance(index, int):
            sources = [sources]
            multichannel_ts = [multichannel_ts]

        assert (
                len(sources) == 1
        ), "sources should be a list"

        sources = preprocess_time_series_stage2(
            copy.deepcopy(sources), self.added_str
        )

        mts_token_ids_list = []
        mts_attention_mask_list = []
        mts_tokenizer_state_list = []
        for ts in multichannel_ts:
            context = torch.stack(ts)
            if isinstance(context, list):
                context = left_pad_and_stack_1D(context)
            assert isinstance(context, torch.Tensor)
            if context.ndim == 1:
                context = context.unsqueeze(0)
            assert context.ndim == 2

            mts_token_ids, mts_attention_mask, mts_tokenizer_state = (
                self.chronos_tokenizer.context_input_transform(context)
            )
            mts_token_ids_list.append(mts_token_ids)
            mts_attention_mask_list.append(mts_attention_mask)
            mts_tokenizer_state_list.append(mts_tokenizer_state)

        if self.tokenizer is None:
            data_dict = dict(
                question=sources[0]["Q"],
                answer=sources[0]["A"],
                cot=sources[0]["cot"],
                ground_truth=sources[0]["ground_truth"],
                mts_token_ids=mts_token_ids_list[0],
                mts_attention_mask=mts_attention_mask_list[0],
                mts_tokenizer_state=mts_tokenizer_state_list[0]
            )
            return data_dict

        data_dict = preprocess(sources, self.tokenizer, self.SYS_INST, self.split, self.preprocess_type, self.preprocess_type_eval)

        data_dict = dict(input_ids=data_dict["input_ids"][0],
                         labels=data_dict["labels"][0],
                         mts_token_ids=mts_token_ids_list[0],
                         mts_attention_mask=mts_attention_mask_list[0],
                         mts_tokenizer_state=mts_tokenizer_state_list[0])
        return data_dict


@dataclass
class DataCollatorForTsTextDatasetStage2(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, mts_token_ids, mts_attention_mask, mts_tokenizer_state = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "mts_token_ids", "mts_attention_mask", "mts_tokenizer_state")
        )

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            mts_token_ids=torch.stack(mts_token_ids),
            mts_attention_mask=torch.stack(mts_attention_mask),
            mts_tokenizer_state=mts_tokenizer_state
        )


def make_ts_text_data_module_stage2(
        tokenizer: transformers.PreTrainedTokenizer, chronos_tokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    data_collator = DataCollatorForTsTextDatasetStage2(tokenizer=tokenizer)
    train_dataset = MultiChannelTimeSeriesDatasetStage2(
        data_path=data_args.data_path, qa_path=data_args.qa_path, tokenizer=tokenizer, chronos_tokenizer=chronos_tokenizer, split="train", data_args=data_args
    )
    eval_dataset = MultiChannelTimeSeriesDatasetStage2(
        data_path=data_args.eval_data_path, qa_path=data_args.eval_qa_path, tokenizer=tokenizer, chronos_tokenizer=chronos_tokenizer, split="eval", data_args=data_args
    )
    return dict(
        train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator
    )


@dataclass
class DataCollatorForTsCLSDatasetStage2(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, mts_token_ids, mts_attention_mask, mts_tokenizer_state = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "mts_token_ids", "mts_attention_mask", "mts_tokenizer_state")
        )

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        return dict(
            input_ids=input_ids,
            labels=torch.tensor(labels),
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            mts_token_ids=torch.stack(mts_token_ids),
            mts_attention_mask=torch.stack(mts_attention_mask),
            mts_tokenizer_state=mts_tokenizer_state
        )


class MultiChannelTimeSeriesCLSDatasetStage2(Dataset):
    def __init__(self, data_path=None, qa_path=None, tokenizer=None, chronos_tokenizer=None, split=None, label2id=None, data_args=None):
        super(MultiChannelTimeSeriesCLSDatasetStage2, self).__init__()
        self.data_path = data_path
        self.qa_path = qa_path
        self.tokenizer = tokenizer
        self.chronos_tokenizer = chronos_tokenizer
        self.split = split
        self.label2id = label2id
        self.preprocess_type = data_args.preprocess_type

        shuffle = data_args.shuffle
        dataset = data_args.dataset


        self.ts_data, self.list_data_dict, self.class_weights = self._flatten_data(shuffle)

        self.data_args = data_args.ts_backbone_config

        self.window_length = len(self.ts_data[0][0])
        self.channel_num = len(self.ts_data[0])
        assert self.channel_num == self.data_args[dataset][
            "channel_num"], "channel_num, data_args.channel_num shape mismatched"

        print(
            f"The dataset size is: {len(self.ts_data)}. Window size: {self.window_length}. Channel num: {self.channel_num}."
        )

        if self.data_args["chronos_model"]["last_token"]:
            added_token = self.data_args["default_ts_token"] * (self.window_length + 1)
        else:
            added_token = self.data_args["default_ts_token"] * self.window_length

        start_tokens_list, end_tokens_list = get_token_list(dataset, self.data_args[dataset], data_args.add_ts_special_token_text)

        added_str = ''
        for start_token, end_token in zip(start_tokens_list, end_tokens_list):
            added_str += start_token + added_token + end_token + '\n'
        self.added_str = added_str

    def _flatten_data(self, shuffle: bool):
        logging.warning(f"Loading {self.split} data...")
        with open(self.data_path, "rb") as f:
            data_file = pickle.load(f)
        with open(self.qa_path, "r") as file:
            qa_file = json.load(file)
        ts_data = []
        qa_dict = []
        label_list = []
        assert len(data_file) == len(qa_file["dataset"])
        for q in qa_file["dataset"]:
            data_idx = q["index"]
            data = data_file[int(data_idx)]
            ts_data.append([torch.from_numpy(data[:, i]).to(torch.float64) for i in range(data.shape[1])])
            answer = q["qa_pair"]["A"]
            try:
                label = int(self.label2id[answer])
            except KeyError:
                raise ValueError(f"Text '{answer}' not found in label2id dictionary")
            label_list.append(label)
            q["qa_pair"]['label'] = label
            qa_dict.append(q["qa_pair"])
        # assert len(ts_data[0]) == 6, "ts_data channel length error"
        # assert len(ts_data[0][0]) == 200, "ts_data length error"
        # assert len(ts_data[0][1]) == 200, "ts_data length error"
        # assert len(ts_data[0][2]) == 200, "ts_data length error"
        assert len(ts_data) == len(qa_dict) == len(label_list), "ts_data, qa_dict, label_list, length not matched"

        class_weights = None
        if self.split == 'train':
            label_series = pd.Series(label_list)
            value_counts = label_series.value_counts(normalize=True)
            class_weights = (1 / value_counts.sort_index()).tolist()
            class_weights = torch.tensor(class_weights)
            class_weights = class_weights / class_weights.sum()
        if shuffle:
            print("Shuffling data...")
            random.seed(RDM_SEED)
            indexes = list(range(len(ts_data)))
            random.shuffle(indexes)
            ts_data = [ts_data[i] for i in indexes]
            qa_dict = [qa_dict[i] for i in indexes]

        # if self.split == "eval":
        #     return ts_data[:100], qa_dict[:100]

        return ts_data, qa_dict, class_weights

    def __len__(self):
        return len(self.ts_data)

    def get_class_weights(self):
        return self.class_weights

    def __getitem__(self, index):
        sources = self.list_data_dict[index]  # {"Q": ..., "A": ...}
        multichannel_ts = self.ts_data[index]  # C * L, 6 * 200

        if isinstance(index, int):
            sources = [sources]
            multichannel_ts = [multichannel_ts]

        assert (
                len(sources) == 1
        ), "sources should be a list"

        sources = preprocess_time_series_CLS_stage2(
            copy.deepcopy(sources)
        )

        mts_token_ids_list = []
        mts_attention_mask_list = []
        mts_tokenizer_state_list = []
        for ts in multichannel_ts:
            context = torch.stack(ts)
            if isinstance(context, list):
                context = left_pad_and_stack_1D(context)
            assert isinstance(context, torch.Tensor)
            if context.ndim == 1:
                context = context.unsqueeze(0)
            assert context.ndim == 2

            mts_token_ids, mts_attention_mask, mts_tokenizer_state = (
                self.chronos_tokenizer.context_input_transform(context)
            )
            mts_token_ids_list.append(mts_token_ids)
            mts_attention_mask_list.append(mts_attention_mask)
            mts_tokenizer_state_list.append(mts_tokenizer_state)

        if self.tokenizer is None:
            data_dict = dict(
                added_str=self.added_str,
                question=sources[0]["Q"],
                smry=sources[0]["smry"],
                trend_text=sources[0]["trend_text"],
                corr_text=sources[0]["corr_text"],
                info_text=sources[0]["info_text"],
                answer=sources[0]["answer"],
                label=sources[0]["label"],
                mts_token_ids=mts_token_ids_list[0],
                mts_attention_mask=mts_attention_mask_list[0],
                mts_tokenizer_state=mts_tokenizer_state_list[0]
            )
            return data_dict

        data_dict = preprocess_cls(sources, self.tokenizer, self.added_str, self.preprocess_type)

        data_dict = dict(input_ids=data_dict["input_ids"][0],
                         input_texts=data_dict["input_texts"][0],
                         labels=sources[0]["label"],
                         answer=sources[0]["answer"],
                         mts_token_ids=mts_token_ids_list[0],
                         mts_attention_mask=mts_attention_mask_list[0],
                         mts_tokenizer_state=mts_tokenizer_state_list[0])

        return data_dict


def make_ts_classification_data_module_stage2(
        tokenizer: transformers.PreTrainedTokenizer, chronos_tokenizer, label2id, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    data_collator = DataCollatorForTsCLSDatasetStage2(tokenizer=tokenizer)
    train_dataset = MultiChannelTimeSeriesCLSDatasetStage2(
        data_path=data_args.data_path, qa_path=data_args.qa_path, tokenizer=tokenizer, chronos_tokenizer=chronos_tokenizer, split="train", label2id=label2id, data_args=data_args
    )
    class_weights = train_dataset.get_class_weights()
    assert class_weights is not None, "class_weights should not be None"

    eval_dataset = MultiChannelTimeSeriesCLSDatasetStage2(
        data_path=data_args.eval_data_path, qa_path=data_args.eval_qa_path, tokenizer=tokenizer, chronos_tokenizer=chronos_tokenizer, split="eval", label2id=label2id, data_args=data_args
    )

    for i in range(2):
        print(f"data example {i}:\nInput Text: {train_dataset[i]['input_texts']}\nLabel: {train_dataset[i]['labels']}\nAnswer: {train_dataset[i]['answer']}\nInput ids: {train_dataset[i]['input_ids']}\n")

    return dict(
        train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator,  class_weights=class_weights
    )