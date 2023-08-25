import torch
from torch import nn
from copy import deepcopy
import pdb
import pytorch_lightning as pl
from torch import optim
import torch.nn.functional as f


def mlp(dim, projection_size = 256, hidden_size = 4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size),
    )


class EncoderWrapper(nn.Module):
    def __init__(
        self,
        model,
        projection_size = 256,
        hidden_size = 4096,
        layer = -1,
    ):
        super().__init__()
        self.model = model
        self.projection_size = projection_size
        self.hidden_size = hidden_size
        self.layer = layer

        self._projector = None
        self._projector_dim = None
        self._encoded = torch.empty(0)
        self._register_hook()

    @property
    def projector(self):
        if self._projector is None:
            self._projector = mlp(
                self._projector_dim, self.projection_size, self.hidden_size
            )
        return self._projector

    # ---------- Methods for registering the forward hook ----------
    # For more info on PyTorch hook, see:
    # https://towardsdatascience.com/how-to-use-pytorch-hooks-5041d777f904
    
    def _hook(self, _, __, output):
        #pdb.set_trace()
        #output = output[0][:,0,:]
        output = output.flatten(start_dim=1)
        if self._projector_dim is None:
            # If we haven't already, measure the output size
            self._projector_dim = output.shape[-1]

        # Project the output to get encodings
        self._encoded = self.projector(output)

    def _register_hook(self):
        if isinstance(self.layer, str):
            layer = dict([*self.model.named_modules()])[self.layer]
        else:
            layer = list(self.model.children())[self.layer]

        layer.register_forward_hook(self._hook)
        
    # ------------------- End hooks methods ----------------------

    def forward(self, input_id, attention_mask):
        # Pass through the model, and collect 'encodings' from our forward hook!
        _ = self.model(input_ids=input_id, attention_mask=attention_mask, return_dict=False)
        return self._encoded


def normalized_mse(x, y):
    x = f.normalize(x, dim=-1)
    y = f.normalize(y, dim=-1)
    return torch.mean(2 - 2 * (x * y).sum(dim=-1))


class BYOL(pl.LightningModule):
    def __init__(
        self,
        model,
        hidden_layer = -1,
        projection_size = 256,
        hidden_size = 4096,
        beta = 0.99,
    ):
        super().__init__()
        self.beta = beta
        self.encoder = EncoderWrapper(
            model, projection_size, hidden_size, layer=hidden_layer
        )
        self.predictor = nn.Linear(projection_size, projection_size, hidden_size)
        self._target = None

        # Perform a single forward pass, which initializes the 'projector' in our 
        # 'EncoderWrapper' layer.
        self.encoder(input_id=torch.zeros(2, 50, dtype=torch.int), attention_mask=torch.zeros(2,1,50, dtype=torch.int))

    def forward(self, x):
        #pdb.set_trace()
        return self.predictor(self.encoder(input_id=x["input_ids"].squeeze(1), attention_mask=x["attention_mask"]))

    @property
    def target(self):
        if self._target is None:
            self._target = deepcopy(self.encoder)
        return self._target

    def update_target(self):
        for p, pt in zip(self.encoder.parameters(), self.target.parameters()):
            pt.data = self.beta * pt.data + (1 - self.beta) * p.data

    # --- Methods required for PyTorch Lightning only! ---

    def configure_optimizers(self):
        optimizer = getattr(optim,"Adam")
        lr = 1e-4
        weight_decay = 1e-6
        return optimizer(self.parameters(), lr=lr, weight_decay=weight_decay)

    def training_step(self, batch, *_):
        x1, x2 = batch[0], batch[1]
        # with torch.no_grad():
        #    x1, x2 = self.augment(x), self.augment(x)

        pred1, pred2 = self.forward(x1), self.forward(x2)
        with torch.no_grad():
            targ1, targ2 = self.target(input_id=x1["input_ids"].squeeze(1), attention_mask=x1["attention_mask"]), self.target(input_id=x2["input_ids"].squeeze(1), attention_mask=x2["attention_mask"])
        loss = (normalized_mse(pred1, targ2) + normalized_mse(pred2, targ1)) / 2

        self.log("train_loss", loss.item())
        # self.update_target()

        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, *_):
        #pdb.set_trace()
        x1, x2 = batch[0], batch[1]
        # x1, x2 = self.augment(x), self.augment(x)
        pred1, pred2 = self.forward(x1), self.forward(x2)
        targ1, targ2 = self.target(input_id=x1["input_ids"].squeeze(1), attention_mask=x1["attention_mask"]), self.target(input_id=x2["input_ids"].squeeze(1), attention_mask=x2["attention_mask"])
        loss = (normalized_mse(pred1, targ2) + normalized_mse(pred2, targ1)) / 2

        return {"loss": loss}

    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        val_loss = sum(x["loss"] for x in outputs) / len(outputs)
        self.log("val_loss", val_loss.item())
