import os
import json
import re
from transformers import AutoTokenizer, AutoConfig
from sensorllm.model.chronos_model import *
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sensorllm.model import *
from sensorllm.data import UniChannelTimeSeriesDataset
from sensorllm.data.utils import generate_chat_template
from sensorllm.utils import disable_torch_init
import warnings
import argparse


warnings.filterwarnings("ignore")

SYS_INST = "A chat between a curious human and an AI assistant. The assistant is given a sequence of N features that represent information extracted from sensor (time-series) readings. The original readings consisted of N data points collected at a sample rate of 100Hz. The assistant's task is to analyze the trends and patterns in the sensor readings by leveraging the encoded information within the features to answer the following specific questions provided by the human."


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model_name_or_path', type=str,
                        default="")
    parser.add_argument('--pt_encoder_backbone_ckpt', type=str,
                        default="")
    parser.add_argument('--tokenize_method', type=str, default="MeanScaleUniformBins")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])

    parser.add_argument('--dataset', type=str, default="usc-had")
    parser.add_argument('--output_file_name', type=str, default="eval.json")
    parser.add_argument('--model_max_length', type=int, default=8192, help='context length during evaluation')
    parser.add_argument('--data_path', type=str, default="",
                        help="Path to the testing data.")
    parser.add_argument('--qa_path', type=str, default="",
                        help="Path to the testing QA data.")
    parser.add_argument('--ignore_qa_types', type=str, nargs='*', default=["sub_trend_no_val"])

    # * data loader, batch_size, shuffle, num_workers
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()
    return args


def load_dataset(data_path, qa_path, chronos_tokenizer):
    print("Loading validation datasets.")
    dataset = UniChannelTimeSeriesDataset(
        data_path=data_path,
        qa_path=qa_path,
        tokenizer=None,  # * load ts and QA
        chronos_tokenizer=chronos_tokenizer,
        data_args=args
    )
    print(f"Example data: {dataset[5]}")
    print("Done!")
    print(dataset)
    return dataset


def custom_collate_fn(batch):
    batch_dict = {
        'question': [],
        'ground_truth': [],
        'type': [],
        'ts_token_ids': [],
        'ts_attention_mask': []
    }

    for item in batch:
        for key in batch_dict:
            batch_dict[key].append(item[key])

    return batch_dict


def get_dataloader(dataset, batch_size, num_workers=2):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=custom_collate_fn)
    return dataloader


def init_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name_or_path)

    # * print the model_name (get the basename)
    print(f'[INFO] Model name: {os.path.basename(model_name)}')

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left"
    )
    model = SensorLLMStage1LlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False, use_cache=False,
                                                              torch_dtype=args.torch_dtype).cuda() 
    model.get_model().load_pt_encoder_backbone_checkpoint(args.pt_encoder_backbone_ckpt,
                                                          tc=args.tokenize_method,
                                                          torch_dtype=args.torch_dtype)
    pt_backbone_config = AutoConfig.from_pretrained(args.pt_encoder_backbone_ckpt)

    assert hasattr(pt_backbone_config, "chronos_config"), "Not a Chronos config file"

    chronos_config = ChronosConfig(**pt_backbone_config.chronos_config)
    chronos_config.tokenizer_class = args.tokenize_method
    chronos_tokenizer = chronos_config.create_tokenizer()

    model.initialize_tokenizer_ts_backbone_config_wo_embedding(tokenizer, dataset=args.dataset)
    model.get_model().load_start_end_tokens(dataset=args.dataset)

    return model, tokenizer, chronos_tokenizer


def generate_outputs(model, tokenizer, inputs, ts_token_ids, ts_attention_mask, do_sample=True, temperature=0.6,
                     top_k=50, max_length=8192, top_p=0.9):
    model.eval()
    model.get_model().pt_encoder_backbone.eval()
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            ts_token_ids=ts_token_ids,
            ts_attention_mask=ts_attention_mask,
            do_sample=do_sample,
            use_cache=False,
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=max_length,
            top_p=top_p,
            eos_token_id=terminators,
            pad_token_id=tokenizer.pad_token_id
        )  # * B, L'
    input_token_len = inputs.input_ids.shape[1]
    n_diff_input_output = (inputs.input_ids != outputs[:, :input_token_len]).sum().item()

    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(outputs[:, input_token_len:], skip_special_tokens=True)
    outputs = [output.strip() for output in outputs]

    return outputs


def start_generation(model, tokenizer, dataloader, output_dir, output_file_name):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, output_file_name)

    results = {"prompt": SYS_INST, "results": []}
    if os.path.exists(output_file):
        # 如果文件已存在，加载现有结果
        with open(output_file, 'r') as f:
            results = json.load(f)

    processed_count = len(results["results"])

    o_i = 0
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        if batch_idx * dataloader.batch_size < processed_count:
            continue
        print(f"start from id {batch_idx}...")

        ts_token_ids = [ts_tensor.cuda() for ts_tensor in batch["ts_token_ids"]]  # * tensor of B, N.
        ts_attention_mask = [ts_tensor.cuda() for ts_tensor in batch["ts_attention_mask"]]

        ground_truths = batch["ground_truth"]  # * list of string
        types = batch["type"]
        questions = batch["question"]  # * list of string

        templated_questions = [generate_chat_template([
            {"role": "system", "content": SYS_INST},
            {"role": "user", "content": q}], bos_token=tokenizer.bos_token, eos_token=tokenizer.eos_token,
            add_generation_prompt=True) for q in
            questions]

        inputs = tokenizer(templated_questions, padding=True, return_tensors="pt").to(model.device)
        outputs = generate_outputs(model, tokenizer, inputs, ts_token_ids,
                                   ts_attention_mask)  # List of str, length is B

        # saving results
        batch_results = []
        for q, gt, output, tp, ts in zip(questions, ground_truths, outputs, types, ts_token_ids):
            result = {
                "questions": q,
                "ground_truth": gt,
                "model_output": output,
                "model_len": len(ts[0]),
                "type": tp
            }
            batch_results.append(result)
            if o_i < 10:
                tqdm.write(f"Type: {tp}\nOutput: {output}\nGround-truth: {gt}\n\n")
                tqdm.write("---------" * 30)
            o_i += 1
        results["results"].extend(batch_results)

        if batch_idx % 10 == 0:  # 每10个批次保存一次
            save_results(results, output_file)

    save_results(results, output_file)
    return results

def save_results(results, output_file):
    temp_file = output_file + '.tmp'
    with open(temp_file, 'w') as f:
        json.dump(results, f, indent=2)

    if os.path.exists(output_file):
        os.replace(temp_file, output_file)
    else:
        os.rename(temp_file, output_file)


def eval(args):
    # * ouptut
    args.output_dir = os.path.join(args.model_name_or_path, "evaluation")
    output_file_path = os.path.join(args.output_dir, args.output_file_name)

    if not os.path.exists(output_file_path):
        # * need inferencing
        model, tokenizer, chronos_tokenizer = init_model(args)
        ts_backbone_config = model.get_model().ts_backbone_config
        args.ts_backbone_config = ts_backbone_config

        dataset = load_dataset(args.data_path, args.qa_path, chronos_tokenizer)
        dataloader = get_dataloader(dataset, args.batch_size, args.num_workers)

        print(f'[INFO] Start generating results for {args.output_file_name}.')
        results = start_generation(model, tokenizer, dataloader, args.output_dir, args.output_file_name)

        del model
        del tokenizer
        torch.cuda.empty_cache()
    else:
        # * directly load the results
        print(f'[INFO] {output_file_path} already exists, directly loading...')
        with open(output_file_path, 'r') as fp:
            results = json.load(fp)
        print(results["results"][:10])


if __name__ == "__main__":
    args = parse_config()
    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    args.torch_dtype = dtype_mapping[args.torch_dtype]

    eval(args)
