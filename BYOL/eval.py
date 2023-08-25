import re

import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from gensim.parsing.preprocessing import STOPWORDS
from rank_eval import Qrels, Run, evaluate
from tqdm import tqdm


a_embed = np.load("./answer_embedding.npy")
q_embed = np.load("./question_embedding.npy")

path = "/home/mty754/dpr/dataset/"

#train_df = pd.read_csv(path+"train.csv")
test_df = pd.read_csv(path+"new_test.csv")

def prepare_Qrels(size):
    true_q_ids = [str(i) for i in range(size)]
    true_doc_ids = [[str(i)] for i in range(size)]
    true_scores = [[100] for _ in range(size)]
    qrels = Qrels()
    qrels.add_multi(q_ids=true_q_ids,doc_ids=true_doc_ids,scores=true_scores)
    return qrels

#qrels = prepare_Qrels(len(train_df))
qrels = prepare_Qrels(len(test_df))

top_k = 5
run = Run()
pred_doc_ids = []
pred_q_ids = []
pred_score = []
for i,q in tqdm(enumerate(q_embed)):
    #import pdb;pdb.set_trace()
    pred_q_ids.append(str(i))

    scores = np.sum((q-a_embed)**2/q.shape[0], axis=1)
    doc_id = np.argsort(scores)[:top_k]
    scores = scores[doc_id]

    pred_score.append(scores.tolist())
    pred_doc_ids.append([str(ids) for ids in doc_id])

run.add_multi(q_ids=pred_q_ids, doc_ids=pred_doc_ids, scores=pred_score)
print(evaluate(qrels, run, ["map@5", "mrr", "ndcg@5", "precision", "recall"]))

#import pdb;pdb.set_trace()
