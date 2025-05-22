from typing import Dict, Sequence
from torch.utils.data import Dataset
import numpy as np
import random
import copy
import json
import pickle
import logging
from dataclasses import dataclass

from sensorllm.data.utils import generate_chat_template, preprocess, get_token_dict
from sensorllm.model.chronos_model import *

import transformers

import torch

IGNORE_INDEX = -100
RDM_SEED = 42


def preprocess_time_series2(
        sources: Sequence[Dict[str, str]],  # [{"Q": "...", "A": "...", "type": ...}]
        channel_names: Sequence[str],
        ts_list: list,
        dataset: str,
        data_args: dict,
) -> Sequence[Dict[str, str]]:
    # assert len(sources) == 6
    ts_token = data_args["default_ts_token"]
    modified_sources = []
    start_tokens_dict, end_tokens_dict = get_token_dict(dataset, data_args)

    for source, channel_name, ts in zip(sources, channel_names, ts_list):
        if data_args["last_token"]:
            added_token = ts_token * (len(ts) + 1)
        else:
            added_token = ts_token * len(ts)
        assert channel_name in list(start_tokens_dict.keys()), f"Start token {channel_name} not found"
        assert channel_name in list(end_tokens_dict.keys()), f"End token {channel_name} not found"
        start_token = start_tokens_dict[channel_name]
        end_token = end_tokens_dict[channel_name]
        modified_q = start_token + added_token + end_token + source["Q"]
        modified_a = source["A"]+"\n\n"+source["summary"]["A"]
        modified_sources.append({"Q": modified_q, "A": modified_a, "type": source["type"]})
    return modified_sources


class UniChannelTimeSeriesDataset(Dataset):
    def __init__(self, data_path=None, qa_path=None, tokenizer=None, chronos_tokenizer=None, split=None, data_args=None):
        """
        data_path: a tensor of shape (N, C, L) where N is the number of multichannel time-series samples,
                   C is the number of channels (6), and L is the sequence length (200).
        qa_path: a list of QA texts corresponding to each channel of each sample.
        """

        super(UniChannelTimeSeriesDataset, self).__init__()
        self.data_path = data_path
        self.qa_path = qa_path
        self.tokenizer = tokenizer
        self.chronos_tokenizer = chronos_tokenizer
        self.split = split

        ignore_qa_types = data_args.ignore_qa_types
        self.dataset = data_args.dataset

        shuffle = data_args.shuffle
        self.data_args = data_args.ts_backbone_config[self.dataset]
        self.data_args["default_ts_token"] = data_args.ts_backbone_config["default_ts_token"]
        self.data_args["last_token"] = data_args.ts_backbone_config["chronos_model"]["last_token"]

        self.SYS_INST = f"A dialogue between a curious researcher and an AI assistant. The AI analyzes a sensor time-series dataset (N points, {self.data_args['sample_rate']}Hz sampling rate) to answer specific questions. This interaction demonstrates the AI's data analysis skills and the potential of human-AI collaboration in interpreting complex data."
        print(f"INSTRUCTION Template: {self.SYS_INST}")
        self.ts_data, self.list_data_dict, self.channel_list = self._flatten_data(ignore_qa_types, shuffle)

        print(
            f"The dataset size is: {len(self.list_data_dict)}."
        )

    def _flatten_data(self, ignore_qa_types: list, shuffle: bool):
        logging.warning("Loading data...")
        with open(self.data_path, "rb") as f:
            data_file = pickle.load(f)
        with open(self.qa_path, "r") as file:
            qa_file = json.load(file)
        qa_dict = []
        ts_data = []
        channel_list = []
        for d in qa_file["dataset"]:
            data_idx = d["index"]
            data = data_file[int(data_idx)]
            if self.dataset in ["usc-had", "uci"]:
                for x_acc, y_acc, z_acc, x_g, y_g, z_g in zip(
                        d["qa_pairs"]["x-axis accelerometer"],
                        d["qa_pairs"]["y-axis accelerometer"],
                        d["qa_pairs"]["z-axis accelerometer"],
                        d["qa_pairs"]["x-axis gyroscope"],
                        d["qa_pairs"]["y-axis gyroscope"],
                        d["qa_pairs"]["z-axis gyroscope"],
                ):
                    assert x_acc["type"] == y_acc["type"] == z_acc["type"] == x_g["type"] == y_g["type"] == z_g[
                        "type"], "QA type values error"
                    # if x_acc["type"] not in ["sub_trend_no_val", "trend_table"]:
                    if x_acc["type"] not in ignore_qa_types:
                        x_acc["summary"] = d['summaries']["x-axis accelerometer"]
                        y_acc["summary"] = d['summaries']["y-axis accelerometer"]
                        z_acc["summary"] = d['summaries']["z-axis accelerometer"]
                        x_g["summary"] = d['summaries']["x-axis gyroscope"]
                        y_g["summary"] = d['summaries']["y-axis gyroscope"]
                        z_g["summary"] = d['summaries']["z-axis gyroscope"]
                        qa_dict.append([x_acc, y_acc, z_acc, x_g, y_g, z_g])
                        ts_data.append(
                            [torch.from_numpy(data[:, i]).to(torch.float64) for i in range(data.shape[1])]
                        )
                        channel_list.extend(["x_acc", "y_acc", "z_acc", "x_g", "y_g", "z_g"])
            elif self.dataset == "capture24":
                for x_acc, y_acc, z_acc in zip(
                        d["qa_pairs"]["x-axis accelerometer"],
                        d["qa_pairs"]["y-axis accelerometer"],
                        d["qa_pairs"]["z-axis accelerometer"]
                ):
                    assert x_acc["type"] == y_acc["type"] == z_acc["type"], "QA type values error"
                    if x_acc["type"] not in ignore_qa_types:
                        x_acc["summary"] = d['summaries']["x-axis accelerometer"]
                        y_acc["summary"] = d['summaries']["y-axis accelerometer"]
                        z_acc["summary"] = d['summaries']["z-axis accelerometer"]
                        qa_dict.append([x_acc, y_acc, z_acc])
                        ts_data.append(
                            [torch.from_numpy(data[:, i]).to(torch.float64) for i in range(data.shape[1])]
                        )
                        channel_list.extend(["x_acc", "y_acc", "z_acc"])
            elif self.dataset == "mhealth":
                for c_acc_x, c_acc_y, c_acc_z, la_acc_x, la_acc_y, la_acc_z, la_gs_x, la_gs_y, la_gs_z, rla_acc_x, rla_acc_y, rla_acc_z, rla_gs_x, rla_gs_y, rla_gs_z in zip(
                        d["qa_pairs"]["chest x-axis accelerometer"], d["qa_pairs"]["chest y-axis accelerometer"], d["qa_pairs"]["chest z-axis accelerometer"],
                        d["qa_pairs"]["left-ankle x-axis accelerometer"], d["qa_pairs"]["left-ankle y-axis accelerometer"], d["qa_pairs"]["left-ankle z-axis accelerometer"],
                        d["qa_pairs"]["left-ankle x-axis gyroscope"], d["qa_pairs"]["left-ankle y-axis gyroscope"], d["qa_pairs"]["left-ankle z-axis gyroscope"],
                        d["qa_pairs"]["right-lower-arm x-axis accelerometer"], d["qa_pairs"]["right-lower-arm y-axis accelerometer"], d["qa_pairs"]["right-lower-arm z-axis accelerometer"],
                        d["qa_pairs"]["right-lower-arm x-axis gyroscope"], d["qa_pairs"]["right-lower-arm y-axis gyroscope"], d["qa_pairs"]["right-lower-arm z-axis gyroscope"]
                ):
                    assert c_acc_x["type"] == c_acc_y["type"] == c_acc_z["type"] == la_acc_x["type"] == la_acc_y[
                        "type"] == la_acc_z["type"] == la_gs_x["type"] == la_gs_y["type"] == la_gs_z[
                        "type"] == rla_acc_x["type"] == rla_acc_y["type"] == rla_acc_z["type"] == rla_gs_x[
                        "type"] == rla_gs_y["type"] == rla_gs_z["type"], "QA type values error"

                    if c_acc_x["type"] not in ignore_qa_types:
                        c_acc_x["summary"] = d['summaries']["chest x-axis accelerometer"]
                        c_acc_y["summary"] = d['summaries']["chest y-axis accelerometer"]
                        c_acc_z["summary"] = d['summaries']["chest z-axis accelerometer"]
                        la_acc_x["summary"] = d['summaries']["left-ankle x-axis accelerometer"]
                        la_acc_y["summary"] = d['summaries']["left-ankle y-axis accelerometer"]
                        la_acc_z["summary"] = d['summaries']["left-ankle x-axis accelerometer"]
                        la_gs_x["summary"] = d['summaries']["left-ankle x-axis gyroscope"]
                        la_gs_y["summary"] = d['summaries']["left-ankle y-axis gyroscope"]
                        la_gs_z["summary"] = d['summaries']["left-ankle z-axis gyroscope"]
                        rla_acc_x["summary"] = d['summaries']["right-lower-arm x-axis accelerometer"]
                        rla_acc_y["summary"] = d['summaries']["right-lower-arm y-axis accelerometer"]
                        rla_acc_z["summary"] = d['summaries']["right-lower-arm z-axis accelerometer"]
                        rla_gs_x["summary"] = d['summaries']["right-lower-arm x-axis gyroscope"]
                        rla_gs_y["summary"] = d['summaries']["right-lower-arm y-axis gyroscope"]
                        rla_gs_z["summary"] = d['summaries']["right-lower-arm z-axis gyroscope"]
                        qa_dict.append([c_acc_x, c_acc_y, c_acc_z, la_acc_x, la_acc_y, la_acc_z, la_gs_x, la_gs_y, la_gs_z, rla_acc_x, rla_acc_y, rla_acc_z, rla_gs_x, rla_gs_y, rla_gs_z])
                        ts_data.append(
                            [torch.from_numpy(data[:, i]).to(torch.float64) for i in range(data.shape[1])]
                        )
                        channel_list.extend(["c_acc_x", "c_acc_y", "c_acc_z", "la_acc_x", "la_acc_y", "la_acc_z", "la_gs_x", "la_gs_y", "la_gs_z", "rla_acc_x", "rla_acc_y", "rla_acc_z", "rla_gs_x", "rla_gs_y", "rla_gs_z"])
            elif self.dataset == "pamap" or self.dataset == "pamap50":
                for acc_hand_x, acc_hand_y, acc_hand_z, gyr_hand_x, gyr_hand_y, gyr_hand_z, mag_hand_x, mag_hand_y, \
                        mag_hand_z, acc_chest_x, acc_chest_y, acc_chest_z, gyr_chest_x, gyr_chest_y, gyr_chest_z, \
                        mag_chest_x, mag_chest_y, mag_chest_z, acc_ankle_x, acc_ankle_y, acc_ankle_z, gyr_ankle_x, \
                        gyr_ankle_y, gyr_ankle_z, mag_ankle_x, mag_ankle_y, mag_ankle_z in zip(
                        d["qa_pairs"]["hand x-axis accelerometer"], d["qa_pairs"]["hand y-axis accelerometer"], d["qa_pairs"]["hand z-axis accelerometer"], d["qa_pairs"]["hand x-axis gyroscope"], d["qa_pairs"]["hand y-axis gyroscope"],
                    d["qa_pairs"]["hand z-axis gyroscope"],d["qa_pairs"]["hand x-axis magnetometer"], d["qa_pairs"]["hand y-axis magnetometer"], d["qa_pairs"]["hand z-axis magnetometer"],d["qa_pairs"]["chest x-axis accelerometer"],
                    d["qa_pairs"]["chest y-axis accelerometer"], d["qa_pairs"]["chest z-axis accelerometer"],d["qa_pairs"]["chest x-axis gyroscope"], d["qa_pairs"]["chest y-axis gyroscope"], d["qa_pairs"]["chest z-axis gyroscope"],
                    d["qa_pairs"]["chest x-axis magnetometer"], d["qa_pairs"]["chest y-axis magnetometer"], d["qa_pairs"]["chest z-axis magnetometer"],d["qa_pairs"]["ankle x-axis accelerometer"], d["qa_pairs"]["ankle y-axis accelerometer"],
                    d["qa_pairs"]["ankle z-axis accelerometer"],d["qa_pairs"]["ankle x-axis gyroscope"], d["qa_pairs"]["ankle y-axis gyroscope"], d["qa_pairs"]["ankle z-axis gyroscope"],d["qa_pairs"]["ankle x-axis magnetometer"],
                    d["qa_pairs"]["ankle y-axis magnetometer"], d["qa_pairs"]["ankle z-axis magnetometer"]

                ):
                    assert acc_hand_x["type"] == acc_hand_y["type"] == acc_hand_z["type"] == gyr_hand_x["type"] == \
                           gyr_hand_y[
                               "type"] == gyr_hand_z["type"] == mag_hand_x["type"] == mag_hand_y["type"] == mag_hand_z[
                               "type"] == acc_chest_x["type"] == acc_chest_y["type"] == acc_chest_z["type"] == \
                           gyr_chest_x[
                               "type"] == gyr_chest_y["type"] == gyr_chest_z["type"] == mag_chest_x["type"] == \
                           mag_chest_y[
                               "type"] == mag_chest_z["type"] == acc_ankle_x["type"] == acc_ankle_y["type"] == \
                           acc_ankle_z[
                               "type"] == gyr_ankle_x["type"] == gyr_ankle_y["type"] == gyr_ankle_z["type"] == \
                           mag_ankle_x[
                               "type"] == mag_ankle_y["type"] == mag_ankle_z["type"], "QA type values error"

                    if acc_hand_x["type"] not in ignore_qa_types:
                        acc_hand_x["summary"] = d['summaries']["hand x-axis accelerometer"]
                        acc_hand_y["summary"] = d['summaries']["hand y-axis accelerometer"]
                        acc_hand_z["summary"] = d['summaries']["hand z-axis accelerometer"]
                        gyr_hand_x["summary"] = d['summaries']["hand x-axis gyroscope"]
                        gyr_hand_y["summary"] = d['summaries']["hand y-axis gyroscope"]
                        gyr_hand_z["summary"] = d['summaries']["hand z-axis gyroscope"]
                        mag_hand_x["summary"] = d['summaries']["hand x-axis magnetometer"]
                        mag_hand_y["summary"] = d['summaries']["hand y-axis magnetometer"]
                        mag_hand_z["summary"] = d['summaries']["hand z-axis magnetometer"]

                        acc_chest_x["summary"] = d['summaries']["chest x-axis accelerometer"]
                        acc_chest_y["summary"] = d['summaries']["chest y-axis accelerometer"]
                        acc_chest_z["summary"] = d['summaries']["chest z-axis accelerometer"]
                        gyr_chest_x["summary"] = d['summaries']["chest x-axis gyroscope"]
                        gyr_chest_y["summary"] = d['summaries']["chest y-axis gyroscope"]
                        gyr_chest_z["summary"] = d['summaries']["chest z-axis gyroscope"]
                        mag_chest_x["summary"] = d['summaries']["chest x-axis magnetometer"]
                        mag_chest_y["summary"] = d['summaries']["chest y-axis magnetometer"]
                        mag_chest_z["summary"] = d['summaries']["chest z-axis magnetometer"]

                        acc_ankle_x["summary"] = d['summaries']["ankle x-axis accelerometer"]
                        acc_ankle_y["summary"] = d['summaries']["ankle y-axis accelerometer"]
                        acc_ankle_z["summary"] = d['summaries']["ankle z-axis accelerometer"]
                        gyr_ankle_x["summary"] = d['summaries']["ankle x-axis gyroscope"]
                        gyr_ankle_y["summary"] = d['summaries']["ankle y-axis gyroscope"]
                        gyr_ankle_z["summary"] = d['summaries']["ankle z-axis gyroscope"]
                        mag_ankle_x["summary"] = d['summaries']["ankle x-axis magnetometer"]
                        mag_ankle_y["summary"] = d['summaries']["ankle y-axis magnetometer"]
                        mag_ankle_z["summary"] = d['summaries']["ankle z-axis magnetometer"]

                        qa_dict.append(
                            [acc_hand_x, acc_hand_y, acc_hand_z, gyr_hand_x, gyr_hand_y, gyr_hand_z, mag_hand_x,
                             mag_hand_y, mag_hand_z,
                             acc_chest_x, acc_chest_y, acc_chest_z, gyr_chest_x, gyr_chest_y, gyr_chest_z, mag_chest_x,
                             mag_chest_y, mag_chest_z,
                             acc_ankle_x, acc_ankle_y, acc_ankle_z, gyr_ankle_x, gyr_ankle_y, gyr_ankle_z, mag_ankle_x,
                             mag_ankle_y, mag_ankle_z])
                        ts_data.append(
                            [torch.from_numpy(data[:, i]).to(torch.float64) for i in range(data.shape[1])]
                        )
                        channel_list.extend([
                            "acc_hand_x", "acc_hand_y", "acc_hand_z",
                            "gyr_hand_x", "gyr_hand_y", "gyr_hand_z",
                            "mag_hand_x", "mag_hand_y", "mag_hand_z",
                            "acc_chest_x", "acc_chest_y", "acc_chest_z",
                            "gyr_chest_x", "gyr_chest_y", "gyr_chest_z",
                            "mag_chest_x", "mag_chest_y", "mag_chest_z",
                            "acc_ankle_x", "acc_ankle_y", "acc_ankle_z",
                            "gyr_ankle_x", "gyr_ankle_y", "gyr_ankle_z",
                            "mag_ankle_x", "mag_ankle_y", "mag_ankle_z"
                        ])
            else:
                raise ValueError(f"Wrong dataset name in _flatten_data: {self.dataset}")


        assert len(ts_data) == len(qa_dict), "ts_data, qa_dict shape mismatched"

        if shuffle:
            print("Shuffling data...")
            random.seed(RDM_SEED)
            indexes = list(range(len(qa_dict)))
            random.shuffle(indexes)

            qa_dict = [qa_dict[i] for i in indexes]
            ts_data = [ts_data[i] for i in indexes]

        qa_dict = [item for sublist in qa_dict for item in sublist]
        ts_data = [item for sublist in ts_data for item in sublist]

        return ts_data, qa_dict, channel_list

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, index):
        sources = self.list_data_dict[index]
        channels = self.channel_list[index]
        ts = self.ts_data[index]  # 1 * L
        if isinstance(index, int):
            sources = [sources]
            channels = [channels]
            ts = [ts]
        assert len(sources) == 1, "sources should be a list"

        sources = preprocess_time_series2(
            copy.deepcopy(sources), copy.deepcopy(channels), copy.deepcopy(ts), self.dataset, self.data_args
        )

        ts_token_ids_list = []
        ts_attention_mask_list = []
        ts_tokenizer_state_list = []
        for context in ts:
            if isinstance(context, list):
                context = left_pad_and_stack_1D(context)
            assert isinstance(context, torch.Tensor)
            if context.ndim == 1:
                context = context.unsqueeze(0)
            assert context.ndim == 2

            ts_token_ids, ts_attention_mask, ts_tokenizer_state = (
                self.chronos_tokenizer.context_input_transform(context)
            )
            ts_token_ids_list.append(ts_token_ids)
            ts_attention_mask_list.append(ts_attention_mask)
            ts_tokenizer_state_list.append(ts_tokenizer_state)

        if self.tokenizer is None:
            data_dict = dict(
                question=sources[0]["Q"],
                ground_truth=sources[0]["A"],
                type=sources[0]["type"],
                ts_token_ids=ts_token_ids_list[0],
                ts_attention_mask=ts_attention_mask_list[0],
                ts_tokenizer_state=ts_tokenizer_state_list[0]
            )
            return data_dict

        data_dict = preprocess(sources, self.tokenizer, self.SYS_INST, self.split, "Q", "Q")

        data_dict = dict(input_ids=data_dict["input_ids"][0],
                         input_texts=sources[0]["Q"],
                         answer=sources[0]["A"],
                         labels=data_dict["labels"][0],
                         ts_token_ids=ts_token_ids_list[0],
                         ts_attention_mask=ts_attention_mask_list[0],
                         ts_tokenizer_state=ts_tokenizer_state_list[0])

        return data_dict


@dataclass
class DataCollatorForTsTextDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, ts_token_ids, ts_attention_mask, ts_tokenizer_state = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "ts_token_ids", "ts_attention_mask", "ts_tokenizer_state")
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
            ts_token_ids=ts_token_ids, # return as list
            ts_attention_mask=ts_attention_mask,
            ts_tokenizer_state=ts_tokenizer_state
        )


def make_ts_text_data_module(
        tokenizer: transformers.PreTrainedTokenizer, chronos_tokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    data_collator = DataCollatorForTsTextDataset(tokenizer=tokenizer)
    train_dataset = UniChannelTimeSeriesDataset(
        data_path=data_args.data_path, qa_path=data_args.qa_path, tokenizer=tokenizer, chronos_tokenizer=chronos_tokenizer, split="train", data_args=data_args
    )
    eval_dataset = UniChannelTimeSeriesDataset(
        data_path=data_args.eval_data_path, qa_path=data_args.eval_qa_path, tokenizer=tokenizer, chronos_tokenizer=chronos_tokenizer, split="train", data_args=data_args
    )
    for i in range(2):
        print(f"data example {i}:\nInput Text: {train_dataset[i]['input_texts']}\nAnswer: {train_dataset[i]['answer']}\nInput ids: {train_dataset[i]['input_ids']}\nTS token ids: {train_dataset[i]['ts_token_ids']}\n")

    return dict(
        train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator
    )
