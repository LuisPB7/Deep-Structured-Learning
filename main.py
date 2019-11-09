import glob
import pandas as pd
import pickle
from model import NLIModel
import torch
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer

torch.manual_seed(0)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

model = NLIModel()

checkpoint_callback = ModelCheckpoint(
    filepath=os.getcwd(),
    save_best_only=False,
    verbose=True,
    monitor='val_loss',
    mode='min',
    prefix=''
)

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.00,
    patience=3,
    verbose=True,
    mode='min'
)

trainer = Trainer(gpus=1, \
                  checkpoint_callback=checkpoint_callback, \
                  #use_amp=True, \
                  show_progress_bar=True, \
                  max_nb_epochs=30, \
                  early_stop_callback=early_stop_callback)

trainer.fit(model)
