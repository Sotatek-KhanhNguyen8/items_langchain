from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings, GPT4AllEmbeddings
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
# import streamlit as st
from pydantic import BaseModel
import uvicorn
# from pyngrok import ngrok
from fastapi import FastAPI
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import AutoModel, AutoTokenizer
# from langchain.tools import tool
# from langchain.agents import initialize_agent, AgentType
# # import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
import ssl
# from pyngrok import ngrok, conf, installer
import os
import csv
import pandas as pd
import shutil
import unicodedata

os.environ["OPENAI_API_KEY"] = ""
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get(""), #you can put the key here directy
    model="gpt-3.5-turbo-16k"
)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def gop_file(file1, file2, output):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Gộp hai DataFrame lại với nhau
    df_combined = pd.concat([df1, df2])

    # Lưu DataFrame kết quả vào một file CSV mới
    df_combined.to_csv(output, index=False)


def csv2txt(csv_link):
    data_text = ''
    with open(csv_link, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Lấy thông tin từ mỗi hàng của file CSV
            PRODUCT_NAME = row[
                'PRODUCT_NAME']  # Thay 'Tên Sản Phẩm' bằng tên cột chứa tên sản phẩm trong file CSV của bạn
            PRODUCT_INFO_ID = row['PRODUCT_INFO_ID']  # Thay 'ID' bằng tên cột chứa ID sản phẩm trong file CSV của bạn
            PRODUCT_CODE = row['PRODUCT_CODE']  # Thay 'Code' bằng tên cột chứa mã code sản phẩm trong file CSV của bạn
            SPECIFICATION_BACKUP = row['SPECIFICATION_BACKUP']
            LINK_SP = row['LINK_SP']
            QUANTITY_SOLD = row['QUANTITY_SOLD']
            # In ra văn bản theo định dạng mong muốn
            s = f"Sản phẩm \"{PRODUCT_NAME}\" có ID là {PRODUCT_INFO_ID}, mã sản phẩm(mã Code) là {PRODUCT_CODE}, thông tin chi tiết về sản phẩm \"{PRODUCT_NAME}\": {SPECIFICATION_BACKUP}, liên kết(Link) của sản phẩm \"{PRODUCT_NAME}\" là {LINK_SP}, số lượng \"{PRODUCT_NAME}\" đã bán là {QUANTITY_SOLD}"
            s = s.replace('\n', ' ')
            s = s.replace('giá: ', f', giá của {PRODUCT_NAME} là ')
            # s = s.replace('  ', ',')
            # s = s.replace('..', ',')
            data_text = data_text + s + '|\n'
            # print(s)
    return data_text


def convert_csv_to_txt(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    csv_files = [f for f in files if f.endswith('.csv')]

    for csv_file in csv_files:
        # Đọc file CSV
        csv_path = os.path.join(directory, csv_file)
        txt_file = csv_file.replace('.csv', '.txt')
        txt_path = os.path.join(directory, txt_file).replace('Bảng tính chưa có tiêu đề', '').replace('-', '').replace(
            ' ', '')
        data_text = csv2txt(csv_path)
        with open(txt_path, "w", encoding='utf-8') as file:
            file.write(data_text)


def convert_and_move_csv_to_txt(source_directory, destination_directory):
    files = [f for f in os.listdir(source_directory) if os.path.isfile(os.path.join(source_directory, f))]
    csv_files = [f for f in files if f.endswith('.csv')]
    os.makedirs(destination_directory, exist_ok=True)
    for csv_file in csv_files:
        # Đọc file CSV
        csv_path = os.path.join(source_directory, csv_file)
        txt_file = csv_file.replace('.csv', '.txt')
        txt_path = os.path.join(destination_directory, txt_file).replace('Bảng tính chưa có tiêu đề', '').replace('-',
                                                                                                             '').replace(
            ' ', '')
        data_text = csv2txt(csv_path)
        with open(txt_path, "w", encoding='utf-8') as file:
            file.write(data_text)
    # os.makedirs(destination_directory, exist_ok=True)
    # files = [f for f in os.listdir(source_directory) if os.path.isfile(os.path.join(source_directory, f))]
    # txt_files = [f for f in files if f.endswith('.txt')]
    # for txt_file in txt_files:
    #     txt_path =  os.path.join(source_directory, txt_file)
    #     dest_path = destination_directory
    #     shutil.move(txt_path, dest_path)


def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def read_all_txt_files(directory):
    txt_contents = {}

    # Lấy danh sách tất cả các file trong thư mục chỉ định
    files = [f for f in os.listdir(directory) if f.endswith('.txt') and os.path.isfile(os.path.join(directory, f))]

    for txt_file in files:
        # Đường dẫn đầy đủ tới file
        file_path = os.path.join(directory, txt_file)

        # Đọc nội dung file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Lưu nội dung vào từ điển với tên file không có phần mở rộng làm khóa
        file_name = os.path.splitext(txt_file)[0]
        txt_contents[file_name] = content

    return txt_contents

def remove_accents(input_str):
    # Normalize the input string to NFD form (Normalization Form Decomposition)
    nfkd_form = unicodedata.normalize('NFD', input_str)
    # Use a list comprehension to filter out the combining characters (accents)
    only_ascii = ''.join([c for c in nfkd_form if not unicodedata.combining(c)])
    return only_ascii.replace('đ','d')

import json
import os

# Hàm để đọc lịch sử trò chuyện dựa trên user_id và conversation_id
def read_history(user_id, conversation_id):
    filename = 'data/history/history.json'

    if not os.path.exists(filename):
        return '',''  # Trả về None nếu tệp không tồn tại

    with open(filename, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    if user_id in data and conversation_id in data[user_id]:
        conversation = data[user_id][conversation_id]
        return conversation['human'], conversation['AI']
    else:
        return '',''  # Trả về None nếu không tìm thấy cuộc trò chuyện

# Hàm để cập nhật lịch sử trò chuyện
def update_history(user_id, conversation_id, question, answer):
    filename = 'data/history/history.json'

    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
    else:
        data = {}

    # Cập nhật dữ liệu mới
    if user_id not in data:
        data[user_id] = {}

    data[user_id][conversation_id] = {
        "human": question,
        "AI": answer
    }

    # Ghi dữ liệu cập nhật vào tệp JSON
    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1001,
    chunk_overlap=0,
    # length_function=len
)
text_splitter_all = CharacterTextSplitter(
    separator="\n",
    chunk_size=1001,
    chunk_overlap=0,
    # length_function=len
)

