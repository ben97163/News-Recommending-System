import json
import os
import numpy as np
from tqdm import tqdm
import bitsandbytes as bnb

import torch
import transformers
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    set_seed,
    Trainer,
    BitsAndBytesConfig,
    LlamaTokenizer
)
from datasets import load_dataset

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)

from utils import get_bnb_config

from pprint import pprint

class DataCollatorForSEQCLS(object):
    def __call__(self, examples):
        return {
            "input_ids": torch.tensor([example["input_ids"] for example in examples]),
            "attention_mask": torch.tensor([example["attention_mask"] for example in examples]),
            "labels": torch.tensor([example["labels"] for example in examples])
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="baffo32/decapoda-research-llama-7B-hf")
    parser.add_argument("--peft_path", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--eval_data", type=str)
    args = parser.parse_args()

    model = AutoModelForSequenceClassification.from_pretrained(
        args.peft_path,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        quantization_config=get_bnb_config()
    )

    model.config.pad_token_id = model.config.eos_token_id

    # model = prepare_model_for_kbit_training(model)

    # lora_config = LoraConfig(
    #     r=8,
    #     lora_alpha=32,
    #     lora_dropout=0.05,
    #     task_type="SEQ_CLS",
    # )

    # model = get_peft_model(model, lora_config)

    tokenizer = LlamaTokenizer.from_pretrained(
        args.model_name_or_path
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    def process_data(example):
        title = example["title"]
        content = example["content"]
        content = content.replace("\n", "")
        content = content.replace("\r", "")
        content = content[:30]

        prompt = f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。預測下列新聞觀看人數。USER: {title}\n{content}"

        tokenized_prompt = tokenizer(prompt, max_length=384, padding="max_length", truncation=True)

        return {
            "input_ids": tokenized_prompt["input_ids"],
            "attention_mask": [1] * len(tokenized_prompt["input_ids"]),
            "labels": [example["view"]],
        }
    
    with open(args.eval_data, "r") as f:
        data = json.load(f)
    
    result = []

    model.eval()

    for i, dic in enumerate(tqdm(data)):
        inputs = process_data(dic)
        inputs = {k: torch.tensor([v]).cuda() for k, v in inputs.items()}
        outputs = model(**inputs)
        predict = outputs.logits.item()
        result.append(predict)

    with open(f"{args.output_dir}/result.json", "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()