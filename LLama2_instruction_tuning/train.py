from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
import argparse
import torch
from datasets import load_dataset, Dataset, DatasetDict
from utils import get_prompt, get_bnb_config
import json
from predict import process
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
def generate_dataset(file_path):
    output_texts = []
    output_answers = []
    # Open the file for reading
    with open(file_path, 'r') as file:
        data = json.load(file)

        # Iterate over each example in the file
        for example in data:
            output_texts.append(example['title'] + '\n' + '你覺得這篇報導的熱門程度如何？')
            if example['label'] == 0:
                output_answers.append('熱門')
            elif example['label'] == 1:
                output_answers.append('普通')
            else:
                output_answers.append('冷門')

    output_dict = {'instruction': output_texts, 'output': output_answers}

    return Dataset.from_dict(output_dict)

def preprocess_function(examples):
    inputs = [get_prompt(doc) for doc in examples["instruction"]]
    
    max_length = 256
    data_size = len(inputs)
    tokenized_instructions = tokenizer(inputs, max_length=max_length, add_special_tokens=False)

    tokenized_outputs = tokenizer(examples["output"], max_length=max_length, add_special_tokens=False)

    for i in range(data_size):

        instruction_input_ids = [tokenizer.bos_token_id] + \
            tokenized_instructions["input_ids"][i]
        output_input_ids = tokenized_outputs["input_ids"][i] + \
            [tokenizer.eos_token_id]
        tokenized_instructions["input_ids"][i] = instruction_input_ids + \
            output_input_ids
        tokenized_instructions["attention_mask"][i] = [
            1] * len(tokenized_instructions["input_ids"][i])

        # tokenized_instructions["input_ids"][i] = torch.tensor(
        #     tokenized_instructions["input_ids"][i][:max_length])
        # tokenized_instructions["attention_mask"][i] = torch.tensor(
        #     tokenized_instructions["attention_mask"][i][:max_length])

        tokenized_outputs["input_ids"][i] = [-100] * len(instruction_input_ids) + output_input_ids
    
    for i in range(data_size):
        sample_input_ids = tokenized_instructions["input_ids"][i]
        label_input_ids = tokenized_outputs["input_ids"][i]

        tokenized_instructions["input_ids"][i] = [tokenizer.pad_token_id] * (max_length - len(sample_input_ids)) + sample_input_ids
        tokenized_instructions["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + tokenized_instructions["attention_mask"][i]

        tokenized_outputs["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids

        tokenized_instructions["input_ids"][i] = torch.tensor(tokenized_instructions["input_ids"][i][:max_length])
        tokenized_instructions["attention_mask"][i] = torch.tensor(tokenized_instructions["attention_mask"][i][:max_length])
        tokenized_outputs["input_ids"][i] = torch.tensor(tokenized_outputs["input_ids"][i][:max_length])

    tokenized_instructions["labels"] = tokenized_outputs["input_ids"]

    return tokenized_instructions


parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_path', type=str, default='Taiwan-LLM-7B-v2.0-chat', help='pretrain model path')
parser.add_argument('--train_file_path', type=str, default='/data1/jerome/ADL/final/train.json', help='pretrain model path')
parser.add_argument('--test_file_path', type=str, default='/data1/jerome/ADL/final/val.json', help='pretrain model path')
parser.add_argument('--output_dir', type=str, default='save_model', help='pretrain model path')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')

args = parser.parse_args()
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=get_bnb_config()
)

tokenizer = AutoTokenizer.from_pretrained(args.model_path)

model = get_peft_model(model, peft_config).to('cuda')
model.print_trainable_parameters()
print("load finish !!!")

train_Dataset = generate_dataset(args.train_file_path)
with open(args.test_file_path) as file:
    test_data = json.load(file)

datasets_dict = {
    "train": train_Dataset
}
dataset = DatasetDict(datasets_dict)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

train_dataset = tokenized_dataset["train"]

batch_size = args.batch_size
train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)

# training argument
num_epochs = 5
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

training_loss = []
eval_acc_history = []
print("start training !!!")

Best_acc = -1

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    steps = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to('cuda') for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        steps += 1
        if steps % 100 == 0:
            training_loss.append(total_loss.item() / 100)
            print(f"Epoch: {epoch + 1} | Step: {steps} | loss: {total_loss.item() / 100}")
            total_loss = 0
        
        if steps % 500 == 0:
            model.eval()
            acc = process(model, tokenizer, test_data)
            eval_acc_history.append(acc)
            print()
            print(f"Epoch: {epoch + 1} | Step: {steps} | acc: {acc}")
            if acc > Best_acc:
                model.save_pretrained(args.output_dir)
                Best_acc = acc
                print(f"finding Best acc: {Best_acc} in epoch: {epoch + 1} step: {steps}")
                with open(os.path.join(args.output_dir, 'acc.txt'),'w') as file:
                    file.write(str(Best_acc))
            model.train()

with open('acc.json','w') as file:
    json.dump(eval_acc_history, file)

with open('loss.json','w') as file:
    json.dump(training_loss, file)