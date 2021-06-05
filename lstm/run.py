import torch.nn as nn
import pytorch_lightning as pl

from datasets import PowerConsumptionDataModule
from module import LSTMRegressor

pl.seed_everything(1)

csv_logger = pl.loggers.csv_logs.CSVLogger('./', name='lstm_logs', version='0')

p = dict(
        seq_len=2,
        batch_size=32,
        criterion=nn.L1Loss(),
        max_epochs=100,
        n_features=2,
        hidden_size=256,
        num_layers=1,
        dropout=0.2,
        learning_rate=0.001,
)

trainer = pl.Trainer(
    max_epochs=p['max_epochs'],
    logger=csv_logger,
    gpus=0,
    progress_bar_refresh_rate=2,
)

model = LSTMRegressor(
        n_features=p['n_features'],
        hidden_size=p['hidden_size'],
        seq_len=p['seq_len'],
        batch_size=p['batch_size'],
        criterion=p['criterion'],
        num_layers=p['num_layers'],
        dropout=p['dropout'],
        learning_rate=p['learning_rate']
)

dm = PowerConsumptionDataModule(
        seq_len=p['seq_len'],
        batch_size=p['batch_size']
)

trainer.fit(model, dm)
metrics = trainer.test(model, datamodule=dm)
