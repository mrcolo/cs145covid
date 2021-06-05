import torch.nn as nn
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from datasets_test import PowerConsumptionDataModule
from module import LSTMRegressor
from sklearn.preprocessing import StandardScaler
import sys

def create_ds(path):
    df = pd.read_csv(
            path, 
            sep=',', 
            parse_dates={'dt' : ['Date']}, 
            infer_datetime_format=True, 
            low_memory=False, 
            na_values=['nan','?'], 
            index_col='dt'
    )
    df = df.sort_values(by=['Province_State', 'dt'])
    states = df['Province_State'].values
    df = df.drop(['Unnamed: 0', 'Recovered', 'Active', 'Incident_Rate',
                    'Total_Test_Results', 'Case_Fatality_Ratio', 'Testing_Rate', 'Province_State'], axis=1)
    X = df.dropna().copy()

    y = X['Confirmed'].shift(-1).ffill()

    preprocessing = StandardScaler()
    preprocessing.fit(X)
    X_test = preprocessing.transform(X)
    y_test = y.values.reshape((-1, 1))
    return X_test, y_test, states


class TimeseriesDataset(Dataset):
    '''
    Custom Dataset subclass. 
    Serves as input to DataLoader to transform X 
      into sequence data using rolling window. 
    DataLoader using this dataset will output batches 
      of `(batch_size, seq_len, n_features)` shape.
    Suitable as an input to RNNs. 
    '''
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 1):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        x = self.X[index:index+self.seq_len]
        y = self.y[index+self.seq_len-1]
        return x, y

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

# trainer = pl.Trainer(
#     max_epochs=p['max_epochs'],
#     logger=csv_logger,
#     gpus=0,
#     progress_bar_refresh_rate=2,
# )

confirmed_model = LSTMRegressor(
        n_features=p['n_features'],
        hidden_size=p['hidden_size'],
        seq_len=p['seq_len'],
        batch_size=p['batch_size'],
        criterion=p['criterion'],
        num_layers=p['num_layers'],
        dropout=p['dropout'],
        learning_rate=p['learning_rate']
)

death_model = LSTMRegressor(
        n_features=p['n_features'],
        hidden_size=p['hidden_size'],
        seq_len=p['seq_len'],
        batch_size=p['batch_size'],
        criterion=p['criterion'],
        num_layers=p['num_layers'],
        dropout=p['dropout'],
        learning_rate=p['learning_rate']
)

confirmed_checkpoint = torch.load(
        "./lstm_logs/0/checkpoints/confirmed.ckpt")
confirmed_model.load_state_dict(confirmed_checkpoint['state_dict'])
confirmed_model.eval()

death_checkpoint = torch.load(
        "./lstm_logs/0/checkpoints/deaths.ckpt")
death_model.load_state_dict(death_checkpoint['state_dict'])
death_model.eval()

x, y, states = create_ds(sys.argv[1])
test_dataset = TimeseriesDataset(x,
                                 y,
                                 seq_len=p['seq_len'])

date = "04-01-2021"
print("ID,Date,Province_State,Confirmed,Deaths", )
for i, (xx, yy) in enumerate(test_dataset):
    if (i % 2 == 0):
        confirmed = confirmed_model.forward(torch.unsqueeze(xx, dim=0)) * 50000
        death = death_model.forward(torch.unsqueeze(xx, dim=0)) * 50000
        curr_state = states[i]
        confirmed_value = confirmed.detach().numpy()[0][0]
        print(f"{date},{curr_state},{int(confirmed)},{int(death)}")
# dm = PowerConsumptionDataModule(
#         seq_len=p['seq_len'],
#         batch_size=p['batch_size']
# )

# metrics = trainer.test(
#     confirmed_model, datamodule=dm)
