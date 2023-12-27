from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    LlamaForSequenceClassification, 
    AutoTokenizer, 
    DataCollatorWithPadding, 
    get_linear_schedule_with_warmup
    )
import argparse
import torch
from datasets import load_dataset, Dataset, DatasetDict
from utils import get_bnb_config
import json
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
import numpy as np


parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', type=str, default='yentinglin/Taiwan-LLM-7B-v2.0-chat', help='pretrain model path')
parser.add_argument('--train_file', type=str, default='../data/train.json', help='pretrain model path')
parser.add_argument('--test_file', type=str, default='../data/val.json', help='pretrain model path')
parser.add_argument('--output_dir', type=str, default='save_model', help='pretrain model path')
parser.add_argument('--batch_size', type=int, default=2, help='batch size for training')
parser.add_argument('--epoch', type=int, default=5, help='epoch for training')

def preprocess(data, max_length=512):
    data_size = len(data)
    title = [x["title"] for x in data]
    content = [(x["content"].replace("\n", "").replace("\r", ""))[:30] for x in data]
    labels = [x["label"] for x in data]

    inputs_ids = []
    attention_mask = []
    for key in batch:
        batch[key] = batch[key].to(device)
    for i in range(data_size):
        text = title[i] + '\n' + content[i]
        tokenized_inputs = tokenizer(text, add_special_tokens=False, max_length=max_length, truncation=True)
        inputs_ids.append(tokenized_inputs["input_ids"])
        attention_mask.append(tokenized_inputs["attention_mask"])


    dataset = {
      "input_ids": inputs_ids,
      "attention_mask": attention_mask,
      "labels": labels
    }

    return dataset

args = parser.parse_args()
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

device = "cuda:0"

peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
model = LlamaForSequenceClassification.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        num_labels=3
)
tokenizer = AutoTokenizer.from_pretrained(args.model)

model.config.pad_token_id = model.config.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

model = get_peft_model(model, peft_config).to(device)
model.print_trainable_parameters()
print("load finish !!!")

with open(args.train_file, "r") as f:
    train_dataset = json.load(f)
with open(args.test_file, "r") as f:
    test_dataset = json.load(f)

train_dataset = Dataset.from_dict(preprocess(train_dataset))
test_dataset = Dataset.from_dict(preprocess(test_dataset))

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)    
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=data_collator)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.00003)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps= len(train_dataloader)*args.epoch
)

best_acc = 0
for epoch in range(args.epoch):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        for key in batch:
            batch[key] = batch[key].to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    
    print('Evaluate...')
    model.eval()
    total_correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dataloader)):
            for key in batch:
                batch[key] = batch[key].to(device)
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            label = batch["labels"].cpu().numpy()

            total_correct_predictions += np.sum(predictions == label)
            total_samples += args.batch_size
        
        accuracy = total_correct_predictions / total_samples
        print(f"[Epoch{epoch}] Accuracy: {accuracy}")
        if(accuracy > best_acc):
            model.save_pretrained(args.output_dir)
            best_acc = accuracy
    