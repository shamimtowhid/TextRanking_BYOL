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
#nltk.download('punkt')

from transformers import BertModel, BertTokenizer, BertConfig


# In[2]:


path = "/home/mty754/dpr/dataset/"


# In[3]:


train_df = pd.read_csv(path+"new_train.csv")
test_df = pd.read_csv(path+"new_test.csv")


# In[4]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[5]:


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


train_dataset = Dataset(train_df)
test_dataset = Dataset(test_df)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)


# In[7]:


class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        config_q = BertConfig()
        config_a = BertConfig()
        self.q_bert = BertModel(config_q)#.from_pretrained('bert-base-uncased')
        self.a_bert = BertModel(config_a)#.from_pretrained('bert-base-uncased')

    def forward(self, q_input_id, q_mask, a_input_id, a_mask):
        #import pdb;pdb.set_trace()
        _, q_pooled_output = self.q_bert(input_ids= q_input_id, attention_mask=q_mask,return_dict=False)
        _, a_pooled_output = self.a_bert(input_ids= a_input_id, attention_mask=a_mask,return_dict=False)
        #matrix = torch.zeros(q_pooled_output.shape[0], a_pooled_output.shape[0], requires_grad = True)
        matrix = torch.matmul(q_pooled_output, torch.t(a_pooled_output))
        return matrix


# In[8]:


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        
    def separate_diagonal_and_sum(self, matrix):
        diagonal = torch.diag(matrix.diag())
        other_elements = torch.exp(matrix - diagonal)
        row_sums = torch.sum(other_elements, dim=1)
        return diagonal, row_sums

    def forward(self, matrix):
        diagonal, row_sums = self.separate_diagonal_and_sum(matrix)
        loss = torch.exp(diagonal)/(torch.exp(diagonal)+row_sums)
        #import pdb;pdb.set_trace()
        return (-1 * torch.log(loss)).mean()


# In[9]:


model = BertClassifier()


# In[10]:


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.cuda.set_device(device)
#criterion = nn.CrossEntropyLoss()
criterion = CustomLoss()
optimizer = Adam(model.parameters(), lr= 1e-5)


# In[11]:


if use_cuda:
    model = model.cuda()
    #criterion = criterion.cuda()


# In[12]:


epochs = 100
for epoch_num in range(epochs):
    total_loss_train = 0
    model.train()
    for question, answer in tqdm(train_dataloader):
        q_mask = question['attention_mask'].to(device)
        q_input_id = question['input_ids'].squeeze(1).to(device)
        a_mask = answer['attention_mask'].to(device)
        a_input_id = answer['input_ids'].squeeze(1).to(device)

        output = model(q_input_id, q_mask, a_input_id, a_mask)

        batch_loss = criterion(output)
        #batch_loss.register_hook(lambda grad: print(grad)) 
        total_loss_train += batch_loss.item()

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        #import pdb;pdb.set_trace()

    total_loss_val = 0
    model.eval()
    with torch.no_grad():
        for question, answer in tqdm(test_dataloader):
            q_mask = question['attention_mask'].to(device)
            q_input_id = question['input_ids'].squeeze(1).to(device)
            a_mask = answer['attention_mask'].to(device)
            a_input_id = answer['input_ids'].squeeze(1).to(device)

            output = model(q_input_id, q_mask, a_input_id, a_mask)

            batch_loss = criterion(output)
            total_loss_val += batch_loss.item()

    print(
        f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_dataloader): .3f} \
        | Val Loss: {total_loss_val / len(test_dataloader): .3f}')

torch.save(model.state_dict(), "./trained_model_100.pth")
