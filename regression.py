import numpy as np
from datetime import  datetime
from sklearn import preprocessing
import pandas as pd
from matplotlib import pyplot as plt
# %matplotlib inline

#Change to 'train_trendency.csv' when not running in google colab
df = pd.read_csv('train_trendency.csv')
df.head()

state_list = [
              'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
              'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia',
              'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
              'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
              'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri',
              'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey',
              'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
              'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
              'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
              'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming',
              ]

df['Date'] = (pd.to_datetime(df['Date']) - datetime(2021, 1, 12)).dt.days

"""
For each state, produce a split of test, train, and validation data
Standardize variables
Currently
  - validation: 29 samples
  - test: 10 samples
  - train: 40 samples
""" 
validation_datas = {}
test_datas = {}
train_datas = {}
states_df = {}
scaled_features = {}

fields_to_drop = ['Unnamed: 0', 'Province_State', 'Recovered', 
                  'Active', 'Incident_Rate', 'Total_Test_Results',
                  'Case_Fatality_Ratio', 'Testing_Rate']

for state in state_list:
  state_df = df[df.loc[:,'Province_State'] == state].copy(deep=True)
  states_df[state] = state_df.copy()
  state_df.drop(fields_to_drop, axis=1, inplace = True)
  state_df.dropna(axis="columns")
  
  continuous_fields = state_df.columns
  scaled_features[state] = {}
  
  for field in continuous_fields:
    mean, std = state_df[field].mean(), state_df[field].std()
    scaled_features[state][field] = [mean, std]
    state_df.loc[:, field] = (state_df[field] - mean)/std

  train_datas[state] = state_df[-40:]
  state_df = state_df[:-40]
  test_datas[state] = state_df[-10:]
  state_df = state_df[:-10]
  validation_datas[state] = state_df

'''
For each target field to predict (Confirmed or Deaths), and for each state
set aside the state respective features/target to train, validate, and test on.
'''
target_fields = ['Confirmed', 'Deaths']
train_features = {}
train_targets = {}
test_features = {}
test_targets = {}
validation_features = {}
validation_targets = {}
for field in target_fields:
  train_features[field] = {}
  train_targets[field] = {}
  test_features[field] = {}
  test_targets[field] = {}
  validation_features[field] = {}
  validation_targets[field] = {}
  for state in state_list:
    train_features[field][state], train_targets[field][state] = train_datas[state].drop(target_fields, axis=1), train_datas[state][target_fields]
    test_features[field][state], test_targets[field][state] = test_datas[state].drop(target_fields, axis=1), test_datas[state][target_fields]
    validation_features[field][state], validation_targets[field][state] = validation_datas[state].drop(target_fields, axis=1), validation_datas[state][target_fields]

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer

l_rate = 0.015
mse_loss = nn.MSELoss(reduction = 'mean')

import time
class Regression(pl.LightningModule):
    def __init__(self, state, field):
        super(Regression, self).__init__()
        self.fc1 = nn.Linear(1, 30)
        self.fc2 = nn.Linear(30, 1)
        self.state = state
        self.field = field
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def train_dataloader(self):
        train_dataset = TensorDataset(torch.tensor(train_features[self.field][self.state].values).float(), torch.tensor(train_targets[self.field][self.state][[self.field]].values).float())
        train_loader = DataLoader(dataset = train_dataset, batch_size = 2)
        return train_loader
        
    def val_dataloader(self):
        validation_dataset = TensorDataset(torch.tensor(validation_features[self.field][self.state].values).float(), torch.tensor(validation_targets[self.field][self.state][[self.field]].values).float())
        validation_loader = DataLoader(dataset = validation_dataset, batch_size = 2)
        return validation_loader
    
    def test_dataloader(self):
        test_dataset = TensorDataset(torch.tensor(test_features[self.field][self.state].values).float(), torch.tensor(test_targets[self.field][self.state][[self.field]].values).float())
        test_loader = DataLoader(dataset = test_dataset, batch_size = 2)
        return test_loader

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=l_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = mse_loss(logits, y)
        # Add logging
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = mse_loss(logits, y)
        return {'val_loss': loss}

    # Define validation epoch end
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        print(logits)
        loss = mse_loss(logits, y)
        correct = torch.sum(logits == y.data)
        predictions_pred[self.field][self.state].append(logits)
        predictions_actual[self.field][self.state].append(y.data)
        return {'test_loss': loss, 'test_correct': correct, 'logits': logits}
    
    # Define test end
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}      
        return {'avg_test_loss': avg_loss, 'log': logs, 'progress_bar': logs }

"""
Create trainers and models for each target field and state
NOTE: This cell seems to take a long time to run for higher epoch,
      but runs faster when the training is split by field into 2 different cells
"""
trainers = {}
models = {}
for field in target_fields:
  trainers[field] = {}
  models[field] = {}
  for state in state_list:
    models[field][state] = Regression(state=state,field=field)
    trainers[field][state] = Trainer(max_epochs=5)
    trainers[field][state].fit(models[field][state])

#New
predictions_pred = {}
predictions_actual = {}
for field in target_fields:
  predictions_pred[field] = {}
  predictions_actual[field] = {}
  for state in state_list:
    predictions_pred[field][state] = []
    predictions_actual[field][state] = []
    trainers[field][state].test(models[field][state])

#New
plots_pred = {}
plots_actual = {}
datesx = []
for field in target_fields:
  plots_pred[field] = {}
  plots_actual[field] = {}

  for state in state_list:
    states_df_copy = states_df[state][:-29]
    datesx = list(states_df_copy[-10:]['Date'])
    datay = list(states_df_copy[-10:][field])
    std = scaled_features[state][field][1]
    mean = scaled_features[state][field][0]

    plots_pred[field][state] = []
    for i in range(len(predictions_pred[field][state])):
        plots_pred[field][state].extend(predictions_pred[field][state][i].T.numpy()[0] * std + mean)

    plots_actual[field][state] = []
    for i in range(len(predictions_actual[field][state])):
        plots_actual[field][state].extend(predictions_actual[field][state][i].T.numpy()[0] * std + mean)

"""
Predicts 30 days of April for each target field and state and places the result into a dict
"""
plot_prediction = {}

predict_dataset = torch.tensor(list(map(lambda x : (x-scaled_features['Alabama']['Date'][0])/scaled_features['Alabama']['Date'][1],range(80,110)))
).float()

for field in target_fields:
  plot_prediction[field] = {}

  for state in state_list:
    plot_prediction[field][state] = []
    std = scaled_features[state][field][1]
    mean = scaled_features[state][field][0]
    april_days = models[field][state](predict_dataset.reshape(30,1))
    for i in april_days:
      plot_prediction[field][state].append(i.detach().numpy()[0]*std + mean)

'''
Creates and formats two tables containing tuples (ID, Confirmed) and (ID, Deaths) respectively
'''
submission_csv = {}
csv_out = {}
for field in target_fields:
  submission_csv[field] = []

  for day in range(0,30):
    for state in plot_prediction[field].keys():
      submission_csv[field].append(plot_prediction[field][state][day])

  csv_out[field] = pd.DataFrame(submission_csv[field], columns=[field])
  csv_out[field] = csv_out[field][[field]]
  csv_out[field].insert(0,'ID',range(0,1500),True)
  csv_out[field] = csv_out[field].round()
  csv_out[field] = csv_out[field].astype('int')

'''
Joins the two tables on ID to create the submission.csv
'''
final_csv = csv_out['Confirmed'].set_index('ID').join(csv_out['Deaths'].set_index('ID'))
final_csv.insert(0,'ID',range(0,1500),True)
final_csv.to_csv('sub.csv', index=False)

'''
Calculate the mape score for the set of test samples
'''
mapes = {}
for field in target_fields:
  mapes[field] = 0
  for state in state_list:
    for i in range(len(plots_pred[field][state])):
      mapes[field] += np.abs((plots_pred[field][state][i] - plots_actual[field][state][i])/plots_actual[field][state][i])
  mapes[field] /= 1500

mape = ((mapes[target_fields[0]] + mapes[target_fields[1]])/2) *100

print(f'Test Set MAPE: {mape}')




