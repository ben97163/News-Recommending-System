
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from peft import PeftModel
from utils import get_prompt, get_bnb_config
import argparse
import json

def process(
    model, tokenizer, data, max_length=2048,
):  
    output = []
    data_size = len(data)
    instructions = [get_prompt(x["title"] + '\n' + '你覺得這篇報導的熱門程度如何？') for x in data]

    # Tokenize data
    tokenized_instructions = tokenizer(instructions, add_special_tokens=False)
    tokenized_instructions['label'] = []

    accs = []
    # Format data
    for i in range(data_size):
        instruction_input_ids = [tokenizer.bos_token_id] + \
            tokenized_instructions["input_ids"][i]
        tokenized_instructions["input_ids"][i] = instruction_input_ids
        tokenized_instructions["attention_mask"][i] = [
            1] * len(tokenized_instructions["input_ids"][i])

        tokenized_instructions["input_ids"][i] = torch.tensor(
            tokenized_instructions["input_ids"][i][:max_length])
        tokenized_instructions["attention_mask"][i] = torch.tensor(
            tokenized_instructions["attention_mask"][i][:max_length])

        tokenized_instructions['label'].append(data[i]['label'])

    for i in tqdm(range(data_size)):
        input_ids = tokenized_instructions["input_ids"][i].unsqueeze(0).to('cuda')
        attn_mask = tokenized_instructions["attention_mask"][i].unsqueeze(0).to('cuda')
        label = input_ids

        with torch.no_grad():
            output_ids = model.generate(input_ids=input_ids)
            text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            if len(text.split('ASSISTANT: ')) == 2:
                predict_text = text.split('ASSISTANT: ')[1]
                if predict_text == '熱門' and tokenized_instructions['label'][i] == 0:
                    accs.append(1)
                elif predict_text == '普通' and tokenized_instructions['label'][i] == 1:
                    accs.append(1)
                elif predict_text == '冷門' and tokenized_instructions['label'][i] == 2:
                    accs.append(1)
                else:
                    accs.append(0)
            else:
                accs.append(0)

    return sum(accs) / len(accs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model_path",
        type=str,
        default="Taiwan-LLM-7B-v2.0-chat",
        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat. If not set, this script will use "
        "the checkpoint from Huggingface (revision = 5073b2bbc1aa5519acdc865e99832857ef47f7c9)."
    )
    parser.add_argument(
        "--peft_path",
        type=str,
        default="save_model",
        help="Path to the saved PEFT checkpoint."
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="data/public_test.json",
        help="Path to test data."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="predict.json",
        help="output path"
    )
    args = parser.parse_args()

    # Load model
    bnb_config = get_bnb_config()

    if args.base_model_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    else:
        model_name = "yentinglin/Taiwan-LLM-7B-v2.0-chat"
        revision = "5073b2bbc1aa5519acdc865e99832857ef47f7c9"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
        )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = PeftModel.from_pretrained(model, args.peft_path)

    with open(args.test_data_path, "r") as f:
        data = json.load(f)
    
    model.eval()

    process(model, tokenizer, data, args.output_path)