#!/usr/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.base.datetools import dates_from_str
from statsmodels.tsa.api import VAR, AutoReg

state_list = ["Alabama", "Alaska", "Arizona", "Arkansas",  "California", "Colorado", "Connecticut",  "Delaware", "Florida", "Georgia",  "Hawaii", "Idaho", "Illinois", "Indiana",  "Iowa", "Kansas", "Kentucky", "Louisiana",  "Maine", "Maryland", "Massachusetts",  "Michigan", "Minnesota", "Mississippi",  "Missouri", "Montana", "Nebraska", "Nevada",  "New Hampshire", "New Jersey", "New Mexico",  "New York", "North Carolina", "North Dakota",  "Ohio", "Oklahoma", "Oregon", "Pennsylvania",  "Rhode Island", "South Carolina",  "South Dakota", "Tennessee", "Texas", "Utah",  "Vermont", "Virginia", "Washington",  "West Virginia", "Wisconsin", "Wyoming"]

df_training = pd.read_csv("./train_trendency.csv")
df_training['Date'] = (pd.to_datetime(df_training['Date']) - datetime(2021, 1, 12)).dt.days
df_vaccine = pd.read_csv("./train_vaccine.csv")
df_vaccine['Date'] = (pd.to_datetime(df_vaccine['Date']) - datetime(2021, 1, 12)).dt.days

df_training = pd.merge(df_training, df_vaccine, on = ['Date', 'Province_State'], how = 'left')
#df_test = df_train[-30:]       #Use last 30 days for testing (March)

state_output = []   #Raw collection of state output data -- needs transformation 

#separate model for each state
for s in state_list:

    df_state = df_training.loc[df_training['Province_State'] == s]

    #Some code adapted from https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/

    predictions = list()

    for col in ['Confirmed', 'Deaths']:
        df_state = df_training.loc[df_training['Province_State'] == s]
        df_state = df_state[[col]]
        df_state = df_state.dropna()

        if col == 'Confirmed':
            lags = 7
        else:
            lags = 5
        model= AutoReg(df_state.values, lags=lags)
        result = model.fit()

        coef = result.params

        history = df_state.values
        history = [history[i] for i in range(len(history))]

        for t in range(30):
            l = len(history)
            lag = [history[i] for i in range(l-lags, l)]

            yhat = coef[0]

            for d in range(lags):
                yhat += coef[d+1] * lag[lags-d-1]
            predictions.append(yhat)
            history.append(yhat)

    reformat = list()
    for i in range(30):
        reformat.append((predictions[i], predictions[i+30]))
    
    state_output.append(reformat)


#re-arrange our output data to match the submission format
output = []
curr_id = 0
for day in range(30):
    for state_num in range(50):
        output.append(state_output[state_num][day])
        curr_id+=1

csv_out = pd.DataFrame(output, columns=['Confirmed', 'Deaths'])

# regenerate the ID column
csv_out.insert(0, "ID", range(0,1500), True)
csv_out = csv_out.round()
csv_out = csv_out.astype('int')

csv_out.to_csv('simpleAR_output.csv', index=False)
