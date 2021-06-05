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

state_output = []   #Raw collection of state output data -- needs transformation 


#Some code was adapted from the examples provided at 
#https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
#and
#https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/

default_cols = ['Confirmed', 'Deaths']      #

#separate model for each state
for s in state_list:
    cols = default_cols

    #New York does not have vaccine data
    if(s == "New York"):
        cols = ['Confirmed', 'Deaths']


    df_state = df_training.loc[df_training['Province_State'] == s]

    df_state = df_state[cols]
    df_state = df_state.dropna()
    model = VAR(df_state)
    
    lag_vars = 4        #Number of previous variables to consider
    results = model.fit(lag_vars)
    #print(results.summary())

    pred = results.forecast(y=df_state.values[-lag_vars:], steps = 30)
    state_output.append(pred)

#re-arrange our output data to match the submission format
output = []
curr_id = 0
for day in range(30):
    for state_num in range(50):
        output.append(state_output[state_num][day])
        curr_id+=1

csv_out = pd.DataFrame(output, columns=cols)
csv_out = csv_out[['Confirmed', 'Deaths']]

# regenerate the ID column
csv_out.insert(0, "ID", range(0,1500), True)
csv_out = csv_out.round()
csv_out = csv_out.astype('int')

csv_out.to_csv('VAR_output.csv', index=False)
#print(csv_out)