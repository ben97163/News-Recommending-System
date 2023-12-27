import time
import datetime
import streamlit as st
import numpy as np
import pandas as pd
import requests
import json
import torch
from tqdm import tqdm
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from utils import get_prompt, get_bnb_config
import argparse
from transformers import (
    LlamaForSequenceClassification,
    AutoTokenizer, 
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)
from bs4 import BeautifulSoup

@st.cache_resource
def load_model():
    print('==== load model ====')
    model = LlamaForSequenceClassification.from_pretrained(
            "yentinglin/Taiwan-LLM-7B-v2.0-chat",
            torch_dtype=torch.bfloat16,
            num_labels=3,
    )
    model = PeftModel.from_pretrained(model, "moose1108/llama_news_lcassification").to(device=device)
    tokenizer = AutoTokenizer.from_pretrained("yentinglin/Taiwan-LLM-7B-v2.0-chat")
    model.score.load_state_dict(torch.load('./score.pt'))
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    print('==== finish loading ====')
    return model, tokenizer

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.92 Safari/537.36',
}

st.title("聯合新聞網新聞推薦系統")
date = st.date_input('Which date of news do you want to read?', value=None)
btn = st.button('Enter')

date_format = "%Y-%m-%d %H:%S"
base_url = "https://udn.com/api/more"

device = 'cuda:4'

def get_day_list():
    page_number = 1
    news_list = []
    sig = 1
    
    while sig: 
        channelId = 1
        cate_id = 99
        type_ = 'breaknews'
        query = f"page={page_number+1}&channelId={channelId}&cate_id={cate_id}&type={type_}"
        news_list_url = base_url + '?' + query
        # print(news_list_url)

        r = requests.get(news_list_url, headers=HEADERS)
        news_data = r.json()
        date_object = datetime.datetime.strptime(str(news_data['lists'][-1]['time']['date']), date_format)

        if date_object.date() > date:
            page_number += 1
            continue
        for i in news_data['lists']:
            date_object = datetime.datetime.strptime(str(i['time']['date']), date_format)
            if date_object.date() == date:
                i['link'] = f"https://udn.com{i['titleLink']}"
                news_list.append(i)
            elif date_object.date() < date:
                sig = 0
                break    
        page_number += 1

    print(len(news_list))
    return news_list

def get_content(news):
    try:
        url = f"https://udn.com{news['titleLink']}"
        r = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(r.text, "html.parser")
        article = soup.find(class_="article-content__editor")
        
        p_tags = article.find_all("p")
        content = ""
        for p in p_tags:
            figures = p.find_all("figure")
            for figure in figures:
                figure.decompose()

        for p in p_tags:
            content += p.text
            content += "\n"

        ret = {
            "title": news["title"],
            "url": url,
            "view": news["view"],
            "time": news["time"],
            "content": content
        }
        return ret
    except:
        return None

def predict(model, tokenizer, title, content, device=device):

    LABEL = ['熱門', '普通', '冷門']

    inputs = title + "\n" + content.replace("\n", "").replace("\r", "")[:30]
    #   print('Predict...')
    model.eval()
    with torch.no_grad():
        tokenized_inputs = tokenizer(inputs, return_tensors="pt").to(device)
        # print(tokenized_inputs)
        outputs = model(**tokenized_inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        # print(LABEL[predictions[0]])
    return LABEL[predictions[0]]

model, tokenizer = load_model()

if btn:
    if date != None:
        with st.spinner('Crawling news, it should take a few seconds...'):
            news_list = get_day_list()

            # result = []
            # for i, news in enumerate(tqdm.tqdm(news_list)):
            #     ret = get_content(news)
            #     # print(ret)
            #     if ret != None:
            #         result.append(ret)

            for i in news_list:
                ans_str = predict(model, tokenizer, i['title'], i['paragraph'][:30])
                if ans_str == '熱門':
                    ans = 0
                elif ans_str == '普通':
                    ans = 1
                elif ans_str == '冷門':
                    ans = 2
                i['ans'] = ans
        st.success('Done!')
        sorted_json_list = sorted(news_list, key=lambda x: x["ans"])

        # st.write(predict(model, tokenizer, '開幕1禮拜就大排長龍！士林夜市開賣大陸傳統點心 網友不酸了：確實特別', 
        #   '\n\r\n近年台灣夜市可以看到不少大陸小吃，例如冰粉、熱奶寶、煎餅果子，就連近期竄紅的「台南幽靈泡麵」，攤販使用'))

        for i in sorted_json_list:
            st.header(i['title'])
            st.write(i['time']['date'])
            st.write(i['link'])
            st.write(i['paragraph'])
            st.image(i['url'])
