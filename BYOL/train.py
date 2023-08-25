import pandas as pd
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader
from model import BYOL
from dataset import Dataset
import numpy as np

from transformers import BertModel, BertConfig

class TargetUpdate(pl.Callback):
    def on_epoch_end(self, trainer, byol):
        byol.update_target()


path = "/home/mty754/dpr/dataset/"

train_df = pd.read_csv(path+"new_train.csv")
test_df = pd.read_csv(path+"new_test.csv")

train_dataset = Dataset(train_df)
test_dataset = Dataset(test_df)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=12)
test_loader = DataLoader(test_dataset, batch_size=32, num_workers=12)

#config = BertConfig()
model = BertModel.from_pretrained("bert-base-uncased")
byol = BYOL(model)

trainer = pl.Trainer(
    max_epochs=100, 
    gpus=-1,
    # Batch size of 2048 matches the BYOL paper
    accumulate_grad_batches=32 // 32,
    weights_summary=None,
    callbacks=[TargetUpdate()]
)

trainer.fit(byol, train_loader, test_loader)
