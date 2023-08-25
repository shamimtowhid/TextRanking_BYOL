import re

import torch
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from gensim.parsing.preprocessing import STOPWORDS

from transformers import BertTokenizer

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
