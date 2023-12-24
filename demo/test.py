import time
import datetime
import streamlit as st
import numpy as np
import pandas as pd
import requests
import json
from transformers import pipeline
from bs4 import BeautifulSoup
import tqdm

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.92 Safari/537.36',
}

st.title("聯合新聞網新聞推薦系統")
date = st.date_input('Which date of news do you want to read?', value=None)
btn = st.button('Enter')

date_format = "%Y-%m-%d %H:%S"
base_url = "https://udn.com/api/more"


def get_day_list():
    page_number = 1
    news_list = []
    sig = 1
    with st.spinner('Crawling news, it should take a few seconds...'):
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
    st.success('Done!')

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

if btn:
    if date != None:
        news_list = get_day_list()

        # result = []
        # for i, news in enumerate(tqdm.tqdm(news_list)):
        #     ret = get_content(news)
        #     # print(ret)
        #     if ret != None:
        #         result.append(ret)

        for i in news_list:
            st.header(i['title'])
            st.write(i['time']['date'])
            st.write(i['link'])
            st.write(i['paragraph'])
            # st.image(i['url'])
