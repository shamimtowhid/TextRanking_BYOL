#!/usr/bin/env python
# coding: utf-8

# In[17]:


import os
import time

import pandas as pd
import numpy as np
from tqdm import tqdm

from hugchat import hugchat
from hugchat.login import Login


# In[3]:


email = ""
pwd = ""


# In[4]:


# Log in to huggingface and grant authorization to huggingchat
sign = Login(email, pwd)
cookies = sign.login()


# In[7]:


chatbot = hugchat.ChatBot(cookies=cookies.get_dict())  # or cookie_path="usercookies/<email>.json"


# In[8]:


#print(chatbot.chat("HI"))


# In[9]:


print(chatbot.chat("Can you answer the following questions for me? The answers should be within 5 to 10 sentences."))


# In[11]:


path = "./dataset/"
train_df = pd.read_csv(path+"train.csv")
#test_df = pd.read_csv(path+"test.csv")


# In[18]:


#len(train_df)


# In[19]:


res = {"long_answer":[], "question":[]}
for question in tqdm(train_df["question"]):
    try:
        res["long_answer"].append(chatbot.chat(question))
        res["question"].append(question)
    except Exception as ex:
        print("error", question)
        res["question"].append(question)
        res["long_answer"].append("Not Found")
        continue
    time.sleep(3)


new_df = pd.DataFrame(res)
new_df.to_csv("./new_train.csv", index=False)
