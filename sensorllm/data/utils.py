from typing import Dict, Sequence
import transformers
import copy

IGNORE_INDEX = -100


def generate_chat_template(messages, bos_token, eos_token, add_generation_prompt=False):
    LLAMA_3_CHAT_TEMPLATE = (
        "{% set loop_messages = messages %}"
        "{% for message in loop_messages %}"
        "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
        "{% if loop.index0 == 0 %}"
        "{% set content = bos_token + content %}"
        "{% endif %}"
        "{{ content }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        "{% endif %}"
    )

    from jinja2 import Template
    template = Template(LLAMA_3_CHAT_TEMPLATE)
    return template.render(messages=messages, bos_token=bos_token, eos_token=eos_token, add_generation_prompt=add_generation_prompt)


def generate_chat_template2(messages, bos_token, eos_token, add_generation_prompt=False):
    LLAMA_3_CHAT_TEMPLATE = (
        "{% set loop_messages = messages %}"
        "{% for message in loop_messages %}"
        "{% if loop.last %}"
        "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim %}"
        "{% else %}"
        "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>'  %}"
        "{% endif %}"
        "{% if loop.index0 == 0 %}"
        "{% set content = bos_token + content %}"
        "{% endif %}"
        "{{ content }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        "{% endif %}"
    )

    from jinja2 import Template
    template = Template(LLAMA_3_CHAT_TEMPLATE)
    return template.render(messages=messages, bos_token=bos_token, eos_token=eos_token, add_generation_prompt=add_generation_prompt)


def _tokenize_fn(
        conversations: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            conv,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for conv in conversations
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def get_token_dict(dataset: str, data_args: dict):
    if dataset in ["usc-had", "uci"]:
        start_tokens_dict = {
            "x_acc": data_args["default_x_acc_start_token"],
            "y_acc": data_args["default_y_acc_start_token"],
            "z_acc": data_args["default_z_acc_start_token"],
            "x_g": data_args["default_x_gyro_start_token"],
            "y_g": data_args["default_y_gyro_start_token"],
            "z_g": data_args["default_z_gyro_start_token"]
        }

        end_tokens_dict = {
            "x_acc": data_args["default_x_acc_end_token"],
            "y_acc": data_args["default_y_acc_end_token"],
            "z_acc": data_args["default_z_acc_end_token"],
            "x_g": data_args["default_x_gyro_end_token"],
            "y_g": data_args["default_y_gyro_end_token"],
            "z_g": data_args["default_z_gyro_end_token"]
        }
    elif dataset == "capture24":
        start_tokens_dict = {
            "x_acc": data_args["default_x_acc_start_token"],
            "y_acc": data_args["default_y_acc_start_token"],
            "z_acc": data_args["default_z_acc_start_token"],
        }

        end_tokens_dict = {
            "x_acc": data_args["default_x_acc_end_token"],
            "y_acc": data_args["default_y_acc_end_token"],
            "z_acc": data_args["default_z_acc_end_token"],
        }
    elif dataset == "mhealth":
        start_tokens_dict = {
            "c_acc_x": data_args["default_chest_x_acc_start_token"],
            "c_acc_y": data_args["default_chest_y_acc_start_token"],
            "c_acc_z": data_args["default_chest_z_acc_start_token"],
            "la_acc_x": data_args["default_left_ankle_x_acc_start_token"],
            "la_acc_y": data_args["default_left_ankle_y_acc_start_token"],
            "la_acc_z": data_args["default_left_ankle_z_acc_start_token"],
            "la_gs_x": data_args["default_left_ankle_x_gyro_start_token"],
            "la_gs_y": data_args["default_left_ankle_y_gyro_start_token"],
            "la_gs_z": data_args["default_left_ankle_z_gyro_start_token"],
            "rla_acc_x": data_args["default_right_lower_arm_x_acc_start_token"],
            "rla_acc_y": data_args["default_right_lower_arm_y_acc_start_token"],
            "rla_acc_z": data_args["default_right_lower_arm_z_acc_start_token"],
            "rla_gs_x": data_args["default_right_lower_arm_x_gyro_start_token"],
            "rla_gs_y": data_args["default_right_lower_arm_y_gyro_start_token"],
            "rla_gs_z": data_args["default_right_lower_arm_z_gyro_start_token"]
        }
        end_tokens_dict = {
            "c_acc_x": data_args["default_chest_x_acc_end_token"],
            "c_acc_y": data_args["default_chest_y_acc_end_token"],
            "c_acc_z": data_args["default_chest_z_acc_end_token"],
            "la_acc_x": data_args["default_left_ankle_x_acc_end_token"],
            "la_acc_y": data_args["default_left_ankle_y_acc_end_token"],
            "la_acc_z": data_args["default_left_ankle_z_acc_end_token"],
            "la_gs_x": data_args["default_left_ankle_x_gyro_end_token"],
            "la_gs_y": data_args["default_left_ankle_y_gyro_end_token"],
            "la_gs_z": data_args["default_left_ankle_z_gyro_end_token"],
            "rla_acc_x": data_args["default_right_lower_arm_x_acc_end_token"],
            "rla_acc_y": data_args["default_right_lower_arm_y_acc_end_token"],
            "rla_acc_z": data_args["default_right_lower_arm_z_acc_end_token"],
            "rla_gs_x": data_args["default_right_lower_arm_x_gyro_end_token"],
            "rla_gs_y": data_args["default_right_lower_arm_y_gyro_end_token"],
            "rla_gs_z": data_args["default_right_lower_arm_z_gyro_end_token"]
        }
    elif dataset == "pamap" or dataset == "pamap50":
        start_tokens_dict = {
            "acc_hand_x": data_args["default_hand_x_acc_start_token"],
            "acc_hand_y": data_args["default_hand_y_acc_start_token"],
            "acc_hand_z": data_args["default_hand_z_acc_start_token"],
            "gyr_hand_x": data_args["default_hand_x_gyro_start_token"],
            "gyr_hand_y": data_args["default_hand_y_gyro_start_token"],
            "gyr_hand_z": data_args["default_hand_z_gyro_start_token"],
            "mag_hand_x": data_args["default_hand_x_mag_start_token"],
            "mag_hand_y": data_args["default_hand_y_mag_start_token"],
            "mag_hand_z": data_args["default_hand_z_mag_start_token"],
            "acc_chest_x": data_args["default_chest_x_acc_start_token"],
            "acc_chest_y": data_args["default_chest_y_acc_start_token"],
            "acc_chest_z": data_args["default_chest_z_acc_start_token"],
            "gyr_chest_x": data_args["default_chest_x_gyro_start_token"],
            "gyr_chest_y": data_args["default_chest_y_gyro_start_token"],
            "gyr_chest_z": data_args["default_chest_z_gyro_start_token"],
            "mag_chest_x": data_args["default_chest_x_mag_start_token"],
            "mag_chest_y": data_args["default_chest_y_mag_start_token"],
            "mag_chest_z": data_args["default_chest_z_mag_start_token"],
            "acc_ankle_x": data_args["default_ankle_x_acc_start_token"],
            "acc_ankle_y": data_args["default_ankle_y_acc_start_token"],
            "acc_ankle_z": data_args["default_ankle_z_acc_start_token"],
            "gyr_ankle_x": data_args["default_ankle_x_gyro_start_token"],
            "gyr_ankle_y": data_args["default_ankle_y_gyro_start_token"],
            "gyr_ankle_z": data_args["default_ankle_z_gyro_start_token"],
            "mag_ankle_x": data_args["default_ankle_x_mag_start_token"],
            "mag_ankle_y": data_args["default_ankle_y_mag_start_token"],
            "mag_ankle_z": data_args["default_ankle_z_mag_start_token"],
        }
        end_tokens_dict = {
            "acc_hand_x": data_args["default_hand_x_acc_end_token"],
            "acc_hand_y": data_args["default_hand_y_acc_end_token"],
            "acc_hand_z": data_args["default_hand_z_acc_end_token"],
            "gyr_hand_x": data_args["default_hand_x_gyro_end_token"],
            "gyr_hand_y": data_args["default_hand_y_gyro_end_token"],
            "gyr_hand_z": data_args["default_hand_z_gyro_end_token"],
            "mag_hand_x": data_args["default_hand_x_mag_end_token"],
            "mag_hand_y": data_args["default_hand_y_mag_end_token"],
            "mag_hand_z": data_args["default_hand_z_mag_end_token"],
            "acc_chest_x": data_args["default_chest_x_acc_end_token"],
            "acc_chest_y": data_args["default_chest_y_acc_end_token"],
            "acc_chest_z": data_args["default_chest_z_acc_end_token"],
            "gyr_chest_x": data_args["default_chest_x_gyro_end_token"],
            "gyr_chest_y": data_args["default_chest_y_gyro_end_token"],
            "gyr_chest_z": data_args["default_chest_z_gyro_end_token"],
            "mag_chest_x": data_args["default_chest_x_mag_end_token"],
            "mag_chest_y": data_args["default_chest_y_mag_end_token"],
            "mag_chest_z": data_args["default_chest_z_mag_end_token"],
            "acc_ankle_x": data_args["default_ankle_x_acc_end_token"],
            "acc_ankle_y": data_args["default_ankle_y_acc_end_token"],
            "acc_ankle_z": data_args["default_ankle_z_acc_end_token"],
            "gyr_ankle_x": data_args["default_ankle_x_gyro_end_token"],
            "gyr_ankle_y": data_args["default_ankle_y_gyro_end_token"],
            "gyr_ankle_z": data_args["default_ankle_z_gyro_end_token"],
            "mag_ankle_x": data_args["default_ankle_x_mag_end_token"],
            "mag_ankle_y": data_args["default_ankle_y_mag_end_token"],
            "mag_ankle_z": data_args["default_ankle_z_mag_end_token"],
        }
    else:
        raise ValueError(f"Wrong dataset name in preprocess_time_series2: {dataset}")
    return start_tokens_dict, end_tokens_dict


def get_token_list(dataset: str, data_args: dict, add_ts_special_token_text: bool):
    if dataset in ["usc-had", "uci"]:
        if add_ts_special_token_text:
            start_tokens_list = [
                "x-axis accelerometer readings: " + data_args["default_x_acc_start_token"],
                "y-axis accelerometer readings: " + data_args["default_y_acc_start_token"],
                "z-axis accelerometer readings: " + data_args["default_z_acc_start_token"],
                "x-axis gyroscope readings: " + data_args["default_x_gyro_start_token"],
                "y-axis gyroscope readings: " + data_args["default_y_gyro_start_token"],
                "z-axis gyroscope readings: " + data_args["default_z_gyro_start_token"]
            ]
        else:
            start_tokens_list = [
                data_args["default_x_acc_start_token"],
                data_args["default_y_acc_start_token"],
                data_args["default_z_acc_start_token"],
                data_args["default_x_gyro_start_token"],
                data_args["default_y_gyro_start_token"],
                data_args["default_z_gyro_start_token"]
            ]

        end_tokens_list = [
            data_args["default_x_acc_end_token"],
            data_args["default_y_acc_end_token"],
            data_args["default_z_acc_end_token"],
            data_args["default_x_gyro_end_token"],
            data_args["default_y_gyro_end_token"],
            data_args["default_z_gyro_end_token"]
        ]
    elif dataset == "capture24":
        if add_ts_special_token_text:
            start_tokens_list = [
                "x-axis accelerometer readings: " + data_args["default_x_acc_start_token"],
                "y-axis accelerometer readings: " + data_args["default_y_acc_start_token"],
                "z-axis accelerometer readings: " + data_args["default_z_acc_start_token"]
            ]
        else:
            start_tokens_list = [
                data_args["default_x_acc_start_token"],
                data_args["default_y_acc_start_token"],
                data_args["default_z_acc_start_token"],
            ]

        end_tokens_list = [
            data_args["default_x_acc_end_token"],
            data_args["default_y_acc_end_token"],
            data_args["default_z_acc_end_token"],
        ]
    elif dataset == "mhealth":
        if add_ts_special_token_text:
            start_tokens_list = [
                "Chest x-axis accelerometer: " + data_args["default_chest_x_acc_start_token"],
                "Chest y-axis accelerometer: " + data_args["default_chest_y_acc_start_token"],
                "Chest z-axis accelerometer: " + data_args["default_chest_z_acc_start_token"],
                "left-ankle x-axis accelerometer: " + data_args["default_left_ankle_x_acc_start_token"],
                "left-ankle y-axis accelerometer: " + data_args["default_left_ankle_y_acc_start_token"],
                "left-ankle z-axis accelerometer: " + data_args["default_left_ankle_z_acc_start_token"],
                "left-ankle x-axis gyroscope: " + data_args["default_left_ankle_x_gyro_start_token"],
                "left-ankle y-axis gyroscope: " + data_args["default_left_ankle_y_gyro_start_token"],
                "left-ankle z-axis gyroscope: " + data_args["default_left_ankle_z_gyro_start_token"],
                "right-lower-arm x-axis accelerometer: " + data_args["default_right_lower_arm_x_acc_start_token"],
                "right-lower-arm y-axis accelerometer: " + data_args["default_right_lower_arm_y_acc_start_token"],
                "right-lower-arm z-axis accelerometer: " + data_args["default_right_lower_arm_z_acc_start_token"],
                "right-lower-arm x-axis gyroscope: " + data_args["default_right_lower_arm_x_gyro_start_token"],
                "right-lower-arm y-axis gyroscope: " + data_args["default_right_lower_arm_y_gyro_start_token"],
                "right-lower-arm z-axis gyroscope: " + data_args["default_right_lower_arm_z_gyro_start_token"]
            ]
        else:
            start_tokens_list = [
                data_args["default_chest_x_acc_start_token"],
                data_args["default_chest_y_acc_start_token"],
                data_args["default_chest_z_acc_start_token"],
                data_args["default_left_ankle_x_acc_start_token"],
                data_args["default_left_ankle_y_acc_start_token"],
                data_args["default_left_ankle_z_acc_start_token"],
                data_args["default_left_ankle_x_gyro_start_token"],
                data_args["default_left_ankle_y_gyro_start_token"],
                data_args["default_left_ankle_z_gyro_start_token"],
                data_args["default_right_lower_arm_x_acc_start_token"],
                data_args["default_right_lower_arm_y_acc_start_token"],
                data_args["default_right_lower_arm_z_acc_start_token"],
                data_args["default_right_lower_arm_x_gyro_start_token"],
                data_args["default_right_lower_arm_y_gyro_start_token"],
                data_args["default_right_lower_arm_z_gyro_start_token"]
            ]
        end_tokens_list = [
            data_args["default_chest_x_acc_end_token"],
            data_args["default_chest_y_acc_end_token"],
            data_args["default_chest_z_acc_end_token"],
            data_args["default_left_ankle_x_acc_end_token"],
            data_args["default_left_ankle_y_acc_end_token"],
            data_args["default_left_ankle_z_acc_end_token"],
            data_args["default_left_ankle_x_gyro_end_token"],
            data_args["default_left_ankle_y_gyro_end_token"],
            data_args["default_left_ankle_z_gyro_end_token"],
            data_args["default_right_lower_arm_x_acc_end_token"],
            data_args["default_right_lower_arm_y_acc_end_token"],
            data_args["default_right_lower_arm_z_acc_end_token"],
            data_args["default_right_lower_arm_x_gyro_end_token"],
            data_args["default_right_lower_arm_y_gyro_end_token"],
            data_args["default_right_lower_arm_z_gyro_end_token"]
        ]
    elif dataset == "pamap" or dataset == "pamap50":
        if add_ts_special_token_text:
            start_tokens_list = [
                "Hand x-axis accelerometer: " + data_args["default_hand_x_acc_start_token"],
                "Hand x-axis accelerometer: " + data_args["default_hand_y_acc_start_token"],
                "Hand x-axis accelerometer: " + data_args["default_hand_z_acc_start_token"],
                "Hand x-axis accelerometer: " + data_args["default_hand_x_gyro_start_token"],
                "Hand x-axis accelerometer: " + data_args["default_hand_y_gyro_start_token"],
                "Hand x-axis accelerometer: " + data_args["default_hand_z_gyro_start_token"],
                "Hand x-axis accelerometer: " + data_args["default_hand_x_mag_start_token"],
                "Hand x-axis accelerometer: " + data_args["default_hand_y_mag_start_token"],
                "Hand x-axis accelerometer: " + data_args["default_hand_z_mag_start_token"],
                "Chest x-axis accelerometer: " + data_args["default_chest_x_acc_start_token"],
                "Chest x-axis accelerometer: " + data_args["default_chest_y_acc_start_token"],
                "Chest x-axis accelerometer: " + data_args["default_chest_z_acc_start_token"],
                "Chest x-axis accelerometer: " + data_args["default_chest_x_gyro_start_token"],
                "Chest x-axis accelerometer: " + data_args["default_chest_y_gyro_start_token"],
                "Chest x-axis accelerometer: " + data_args["default_chest_z_gyro_start_token"],
                "Chest x-axis accelerometer: " + data_args["default_chest_x_mag_start_token"],
                "Chest x-axis accelerometer: " + data_args["default_chest_y_mag_start_token"],
                "Chest x-axis accelerometer: " + data_args["default_chest_z_mag_start_token"],
                "Ankle x-axis accelerometer: " + data_args["default_ankle_x_acc_start_token"],
                "Ankle x-axis accelerometer: " + data_args["default_ankle_y_acc_start_token"],
                "Ankle x-axis accelerometer: " + data_args["default_ankle_z_acc_start_token"],
                "Ankle x-axis accelerometer: " + data_args["default_ankle_x_gyro_start_token"],
                "Ankle x-axis accelerometer: " + data_args["default_ankle_y_gyro_start_token"],
                "Ankle x-axis accelerometer: " + data_args["default_ankle_z_gyro_start_token"],
                "Ankle x-axis accelerometer: " + data_args["default_ankle_x_mag_start_token"],
                "Ankle x-axis accelerometer: " + data_args["default_ankle_y_mag_start_token"],
                "Ankle x-axis accelerometer: " + data_args["default_ankle_z_mag_start_token"]
            ]
        else:
            start_tokens_list = [
                data_args["default_hand_x_acc_start_token"],
                data_args["default_hand_y_acc_start_token"],
                data_args["default_hand_z_acc_start_token"],
                data_args["default_hand_x_gyro_start_token"],
                data_args["default_hand_y_gyro_start_token"],
                data_args["default_hand_z_gyro_start_token"],
                data_args["default_hand_x_mag_start_token"],
                data_args["default_hand_y_mag_start_token"],
                data_args["default_hand_z_mag_start_token"],
                data_args["default_chest_x_acc_start_token"],
                data_args["default_chest_y_acc_start_token"],
                data_args["default_chest_z_acc_start_token"],
                data_args["default_chest_x_gyro_start_token"],
                data_args["default_chest_y_gyro_start_token"],
                data_args["default_chest_z_gyro_start_token"],
                data_args["default_chest_x_mag_start_token"],
                data_args["default_chest_y_mag_start_token"],
                data_args["default_chest_z_mag_start_token"],
                data_args["default_ankle_x_acc_start_token"],
                data_args["default_ankle_y_acc_start_token"],
                data_args["default_ankle_z_acc_start_token"],
                data_args["default_ankle_x_gyro_start_token"],
                data_args["default_ankle_y_gyro_start_token"],
                data_args["default_ankle_z_gyro_start_token"],
                data_args["default_ankle_x_mag_start_token"],
                data_args["default_ankle_y_mag_start_token"],
                data_args["default_ankle_z_mag_start_token"]
            ]
        end_tokens_list = [
            data_args["default_hand_x_acc_end_token"],
            data_args["default_hand_y_acc_end_token"],
            data_args["default_hand_z_acc_end_token"],
            data_args["default_hand_x_gyro_end_token"],
            data_args["default_hand_y_gyro_end_token"],
            data_args["default_hand_z_gyro_end_token"],
            data_args["default_hand_x_mag_end_token"],
            data_args["default_hand_y_mag_end_token"],
            data_args["default_hand_z_mag_end_token"],
            data_args["default_chest_x_acc_end_token"],
            data_args["default_chest_y_acc_end_token"],
            data_args["default_chest_z_acc_end_token"],
            data_args["default_chest_x_gyro_end_token"],
            data_args["default_chest_y_gyro_end_token"],
            data_args["default_chest_z_gyro_end_token"],
            data_args["default_chest_x_mag_end_token"],
            data_args["default_chest_y_mag_end_token"],
            data_args["default_chest_z_mag_end_token"],
            data_args["default_ankle_x_acc_end_token"],
            data_args["default_ankle_y_acc_end_token"],
            data_args["default_ankle_z_acc_end_token"],
            data_args["default_ankle_x_gyro_end_token"],
            data_args["default_ankle_y_gyro_end_token"],
            data_args["default_ankle_z_gyro_end_token"],
            data_args["default_ankle_x_mag_end_token"],
            data_args["default_ankle_y_mag_end_token"],
            data_args["default_ankle_z_mag_end_token"]
        ]
    else:
        raise ValueError(f"Wrong dataset name in preprocess_time_series2: {dataset}")
    return start_tokens_list, end_tokens_list


def preprocess(
        sources: Sequence[Dict[str, str]],
        tokenizer: transformers.PreTrainedTokenizer,
        SYS_INST: str,
        split: str,
        preprocess_type: str,
        preprocess_type_eval: str,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [generate_chat_template([
        {"role": "system", "content": SYS_INST},
        {"role": "user", "content": s["Q"]},
        {"role": "assistant", "content": s["A"]}], bos_token=tokenizer.bos_token, eos_token=tokenizer.eos_token, add_generation_prompt=False) for s in
        sources]

    if split == 'train':
        if preprocess_type == "Q":
            sources_q = [generate_chat_template([
                {"role": "system", "content": SYS_INST},
                {"role": "user", "content": s["Q"]}], bos_token=tokenizer.bos_token, eos_token=tokenizer.eos_token, add_generation_prompt=True) for s in
                sources]
        else:
            assert preprocess_type == "Q+cot"
            sources_q = [generate_chat_template2([
                {"role": "system", "content": SYS_INST},
                {"role": "user", "content": s["Q"]},
                {"role": "assistant", "content": s["cot"]}], bos_token=tokenizer.bos_token,
                eos_token=tokenizer.eos_token,
                add_generation_prompt=False) for s in
                sources]
    else:
        assert split == 'eval'
        if preprocess_type_eval == "Q":
            sources_q = [generate_chat_template([
                {"role": "system", "content": SYS_INST},
                {"role": "user", "content": s["Q"]}], bos_token=tokenizer.bos_token, eos_token=tokenizer.eos_token, add_generation_prompt=True) for s in
                sources]
        else:
            assert preprocess_type_eval == "Q+cot"
            sources_q = [generate_chat_template2([
                {"role": "system", "content": SYS_INST},
                {"role": "user", "content": s["Q"]},
                {"role": "assistant", "content": s["cot"]}], bos_token=tokenizer.bos_token,
                eos_token=tokenizer.eos_token,
                add_generation_prompt=False) for s in
                sources]

    examples_tokenized = _tokenize_fn(examples, tokenizer)
    sources_tokenized = _tokenize_fn(sources_q, tokenizer)

    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def preprocess_cls(
        sources: Sequence[Dict[str, str]],
        tokenizer: transformers.PreTrainedTokenizer,
        added_str: str,
        preprocess_type: str
) -> Dict:
    """Preprocess the data by tokenizing."""

    if preprocess_type == "smry":
        inputs = [added_str + '\n' + s["smry"] for s in sources]
    elif preprocess_type == "trend":
        inputs = [added_str + '\n' + s["trend_text"] for s in sources]
    elif preprocess_type == "corr":
        inputs = [added_str + '\n' + s["corr_text"] for s in sources]
    elif preprocess_type == "none":
        inputs = [added_str for _ in sources]
    elif preprocess_type == "smry+Q":
        inputs = [added_str + '\n' + s["smry"] + '\n' + s["Q"] for s in sources]
    elif preprocess_type == "smry+meta":
        inputs = [added_str + '\n' + s["smry"] + '\n' + s["info_text"] for s in sources]
    elif preprocess_type == "smry+meta+Q":
        inputs = [added_str + '\n' + s["smry"] + '\n' + s["info_text"] + '\n' + s["Q"] for s in sources]
    elif preprocess_type == "smry+corr":
        inputs = [added_str + '\n' + s["smry"] + '\n' + s["corr_text"] for s in sources]
    elif preprocess_type == "smry+corr+Q":
        inputs = [added_str + '\n' + s["smry"] + '\n' + s["corr_text"] + '\n' + s["Q"] for s in sources]
    elif preprocess_type == "smry+trend+corr":
        inputs = [added_str + '\n' + s["smry"] + '\n' + s["trend_text"] + '\n' + s["corr_text"] for s in sources]
    elif preprocess_type == "smry+trend+corr+Q":
        inputs = [added_str + '\n' + s["smry"] + '\n' + s["trend_text"] + '\n' + s["corr_text"] + '\n' + s["Q"] for s in sources]
    elif preprocess_type == "smry+trend+Q":
        inputs = [added_str + '\n' + s["smry"] + '\n' + s["trend_text"] + '\n' + s["Q"] for s in sources]
    else:
        assert preprocess_type == "smry+trend", f"Undefined preprocess_type {preprocess_type}"
        inputs = [added_str + '\n' + s["smry"] + '\n' + s["trend_text"] for s in sources]

    inputs_tokenized = _tokenize_fn(inputs, tokenizer)

    input_ids = inputs_tokenized["input_ids"]
    return dict(input_ids=input_ids, input_texts=inputs)