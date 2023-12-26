import json
import os
import numpy as np
from tqdm import tqdm
import bitsandbytes as bnb

import torch
import transformers
import argparse
from transformers import (
    AutoModelForSequenceClassification,
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


def predict(mode, tokenizer, title, content, url, device="cuda"):
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

    model.eval()

    inputs = process_data({"title": title, "content": content})
    inputs = {k: torch.tensor([v]).to(device) for k, v in inputs.items()}

    outputs = model(**inputs)

    prediction = outputs.logits.item()

    return {"url": url, "title": title, "content": content, "predict_views": prediction}

if __name__ == "__main__":
    model = AutoModelForSequenceClassification.from_pretrained(
        peft_path,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        quantization_config=get_bnb_config()
    )

    model.config.pad_token_id = model.config.eos_token_id

    tokenizer = LlamaTokenizer.from_pretrained(
        model_name_or_path
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    prediction = predict(
        model, 
        tokenizer, 
        "特色茶＋招牌粉粿！「八曜和茶×一沐日」聯名了", 
        "超人氣的連鎖手搖飲「八曜和茶」、「一沐日」聯名了！自12月22日起，兩間手搖飲共同推出期間限定的「八曜一沐日」聯名活動，將推出結合雙品牌特色元素的「粉粿柚香307」與「粉粿舞伎406奶茶」，同時還有保溫杯、保溫袋，以及只送不賣的「守護愛の斗篷」及「杜邦單肩袋」等一系列週邊商品同步登場。\n\n\n\r\n來自高雄的「八曜和茶」在手搖飲中加入自然素材的概念，主打穀、麥、草本元素的止渴茶飲。台中起家的「一沐日」，則擅長推出具有台灣特色的茶品與配料，兩個品牌各有不同特色，且都有死忠的支持者。\n\n\n\r\n本次雙品牌聯名的主題，標榜以「愛」為出發點。其中「粉粿柚香307」使用八曜和茶的極上柚香307融合新鮮完熟甜柚，再配上一沐日招牌的手工粉粿，藉由炭焙烏龍茶香、紅柚果香以及粉粿口感的結合，帶來豐富的層次變化，每杯79元。另款「粉粿舞伎406奶茶」則是八曜的舞伎406紅茶搭配香醇焙煎厚奶，以及一沐日的紅茶粉粿，每杯79元。\n\n\n\r\n除了聯名飲品外，還推出「星星象印保溫杯」、「愛の保溫袋」、「守護愛の斗篷」及「杜邦單肩袋」。聯名款「星星象印保溫杯」有代表八曜的白色款，以及代表一沐日的綠色款，售價1,290元。「愛の保溫袋」以此次聯名的主視覺設計出發，加上豐富的品牌標語，每個109元，加購價69元。\n\n\n\r\nVIP限定周邊「守護愛の斗篷」共有黃金傳說版與星星閃耀版2款。至於「愛の限定-杜邦單肩袋」以聯名元素作為主要視覺。「守護愛の斗篷」及「杜邦單肩袋」將透過抽獎活動送出，只送不賣。\n", 
        "https://udn.com/news/story/7270/7647572?from=udn-ch1_breaknews-1-0-news"
    )

    pprint(prediction)