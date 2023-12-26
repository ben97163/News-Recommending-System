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
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--eval_data", type=str)
    args = parser.parse_args()

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        quantization_config=get_bnb_config()
    )

    model.config.pad_token_id = model.config.eos_token_id

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        task_type="SEQ_CLS",
    )

    model = get_peft_model(model, lora_config)

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
    
    dataset = load_dataset("json", data_files={"train": args.train_data, "eval": args.eval_data})
    train_dataset = dataset["train"].shuffle().map(process_data)
    eval_dataset = dataset["eval"].map(process_data)

    trainer = Trainer(
        model=model,
        args=transformers.TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=3e-4,
            evaluation_strategy="steps",
            save_strategy="steps",
            logging_steps=20,
            save_steps=100,
            save_total_limit=10,
            num_train_epochs=args.epoch,
            report_to=[],
            remove_unused_columns=False,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSEQCLS()
    )
    
    trainer.train()

    trainer.model.save_pretrained(args.output_dir)

    torch.save(trainer.model.score.state_dict(), "./score.ckpt")
    
if __name__ == "__main__":
    main()