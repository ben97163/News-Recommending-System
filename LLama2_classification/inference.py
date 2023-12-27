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
from peft import PeftModel
import os
from torch.utils.data import DataLoader
import numpy as np


def predict(model, tokenizer, title, content, device='cuda:3'):

  LABEL = ['熱門', '普通', '冷門']

  inputs = title + "\n" + content.replace("\n", "").replace("\r", "")[:30]
  print('Predict...')
  model.eval()
  with torch.no_grad():
    tokenized_inputs = tokenizer(inputs, return_tensors="pt").to(device)
    print(tokenized_inputs)
    outputs = model(**tokenized_inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).cpu().numpy()
    print(LABEL[predictions[0]])
  return LABEL[predictions[0]]
    

    
if __name__ == "__main__":
  model_name = 'yentinglin/Taiwan-LLM-7B-v2.0-chat'
  model = LlamaForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            num_labels=3
  )
  model = PeftModel.from_pretrained(model, './save_model').to('cuda:3')
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  model.config.pad_token_id = model.config.eos_token_id
  tokenizer.pad_token = tokenizer.eos_token

  predict(model, tokenizer, '開幕1禮拜就大排長龍！士林夜市開賣大陸傳統點心 網友不酸了：確實特別', 
          '\n\r\n近年台灣夜市可以看到不少大陸小吃，例如冰粉、熱奶寶、煎餅果子，就連近期竄紅的「台南幽靈泡麵」，攤販使用')
          
