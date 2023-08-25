#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re

import torch
import pandas as pd
import numpy as np

import nltk
from rank_bm25 import *
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from gensim.parsing.preprocessing import STOPWORDS
from rank_eval import Qrels, Run, evaluate
from tqdm import tqdm
from torch.optim import Adam
from torch import nn


from transformers import BertModel, BertTokenizer, BertConfig


# In[2]:


path = "/home/mty754/dpr/dataset/"


# In[3]:


#train_df = pd.read_csv(path+"train.csv")
test_df = pd.read_csv(path+"new_test.csv")


# In[4]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.q_texts = self.process_text(df["question"])
        self.a_texts = self.process_text(df["final_answer"])
            
    def process_text(self, series):
        ps = PorterStemmer()
        texts = []
        for item in series:
            removed_special_ch = re.sub("[^0-9a-zA-Z]"," ",item)
            text_tokens = word_tokenize(removed_special_ch)
            removed_stop_words = " ".join([word for word in text_tokens if not word in STOPWORDS])
            stemmed_words = " ".join([ps.stem(w) for w in removed_stop_words.split()])
            
            texts.append(tokenizer(stemmed_words, padding='max_length',
                      max_length = 300, truncation=True, return_tensors="pt"))
        return texts

    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        return self.q_texts[idx], self.a_texts[idx]


# In[6]:
BATCH_SIZE = 32

#train_dataset = Dataset(train_df)
test_dataset = Dataset(test_df)

#train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# In[7]:


class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        config_q = BertConfig()
        config_a = BertConfig()
        self.q_bert = BertModel(config_q)#.from_pretrained('bert-base-uncased')
        self.a_bert = BertModel(config_a)#.from_pretrained('bert-base-uncased')

    def forward(self, q_input_id, q_mask, a_input_id, a_mask):
        _, q_pooled_output = self.q_bert(input_ids= q_input_id, attention_mask=q_mask,return_dict=False)
        _, a_pooled_output = self.a_bert(input_ids= a_input_id, attention_mask=a_mask,return_dict=False)

        return q_pooled_output, a_pooled_output


model = BertClassifier()
model.load_state_dict(torch.load("./trained_model_100.pth"))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.cuda.set_device(device)

if use_cuda:
    model = model.cuda()

model.eval()
answer_embedding = torch.zeros((len(test_df), 768))
question_embedding = torch.zeros((len(test_df), 768))

with torch.no_grad():
    for i, (question, answer) in tqdm(enumerate(test_dataloader)):
        q_mask = question['attention_mask'].to(device)
        q_input_id = question['input_ids'].squeeze(1).to(device)
        a_mask = answer['attention_mask'].to(device)
        a_input_id = answer['input_ids'].squeeze(1).to(device)

        q_out, a_out = model(q_input_id, q_mask, a_input_id, a_mask)
        answer_embedding[(i*BATCH_SIZE):(i*BATCH_SIZE)+BATCH_SIZE,:] = a_out
        question_embedding[(i*BATCH_SIZE):(i*BATCH_SIZE)+BATCH_SIZE,:] = q_out

answer_numpy = answer_embedding.numpy()
question_numpy = question_embedding.numpy()

np.save("question_embedding.npy", question_numpy)
np.save("answer_embedding.npy", answer_numpy)
