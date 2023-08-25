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
from tqdm import tqdm
from torch import nn
from transformers import BertModel, BertTokenizer, BertConfig

from model import BYOL

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


config = BertConfig()
bert_model = BertModel(config)
model = BYOL(bert_model)
checkpoint = torch.load("./lightning_logs/version_3/checkpoints/epoch=46.ckpt")

old_state_dict = checkpoint["state_dict"]

new_state_dict = {}
for k in model.state_dict().keys():
    new_state_dict[k] = old_state_dict[k]


model.load_state_dict(new_state_dict)

#import pdb;pdb.set_trace()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.cuda.set_device(device)

if use_cuda:
    model = model.cuda()

model.eval()
#answer_embedding = torch.zeros((len(train_df), 256))
#question_embedding = torch.zeros((len(train_df), 256))
answer_embedding = torch.zeros((len(test_df), 256))
question_embedding = torch.zeros((len(test_df), 256))

with torch.no_grad():
    #for i, (question, answer) in tqdm(enumerate(train_dataloader)):
    for i, (question, answer) in tqdm(enumerate(test_dataloader)):
        question = question.to(device)
        answer = answer.to(device)
        q_out = model(question)
        a_out = model(answer)

        answer_embedding[(i*BATCH_SIZE):(i*BATCH_SIZE)+BATCH_SIZE,:] = a_out
        question_embedding[(i*BATCH_SIZE):(i*BATCH_SIZE)+BATCH_SIZE,:] = q_out

answer_numpy = answer_embedding.numpy()
question_numpy = question_embedding.numpy()

np.save("question_embedding.npy", question_numpy)
np.save("answer_embedding.npy", answer_numpy)
