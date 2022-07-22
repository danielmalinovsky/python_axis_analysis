# %% Libraries
import re
from tempfile import tempdir
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.stats.api as sms
from statsmodels.stats.stattools import durbin_watson
#from yahoofinancials import YahooFinancials

from statistics import stdev

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
from statsmodels.tsa.ar_model import AutoReg

from sklearn.metrics import mean_squared_error

from math import sqrt

from pandas import concat

# %%
# User defined functions

def yields_preprocessing(data, clustering = False):

    data['hilo_diff'] = 0
    data['diff'] = 0

    for i in range(1,len(data['Date'])):
        if (data['Close'].iloc[i,] - data['Close'].iloc[i-1,]) < 0:
            data['hilo_diff'].iloc[i,] = (data['Low'].iloc[i,] / data['High'].iloc[i-1,]) - 1
        else:
            data['hilo_diff'].iloc[i,] = (data['High'].iloc[i,] / data['Low'].iloc[i-1,]) - 1

    for i in range(1,len(data['Date'])):
        data['diff'].iloc[i,] = (data['Close'].iloc[i,] / data['Close'].iloc[i-1,]) - 1

    output = data
    
    # XY analysis time periods clustering

    if clustering == True:

        data['day_buffer'] = 0
        data['hilo_diff_buffer'] = 0
        data_xy = pd.DataFrame(columns=data.columns)

        for index, row in data.iterrows():
            if np.logical_and(data['hilo_diff'].iloc[index] > 0, data['hilo_diff'].iloc[index - 1] < 0):
                data['day_buffer'].iloc[index] = 1
                data['hilo_diff_buffer'].iloc[index] = data['hilo_diff'].iloc[index]
                data_xy = data_xy.append(data.iloc[index-1], ignore_index=True)
            elif np.logical_and(data['hilo_diff'].iloc[index] > 0, data['hilo_diff'].iloc[index - 1] > 0):
                data['day_buffer'].iloc[index] = data['day_buffer'].iloc[index - 1] + 1
                data['hilo_diff_buffer'].iloc[index] = data['hilo_diff_buffer'].iloc[index - 1] + data['hilo_diff'].iloc[index]
            elif np.logical_and(data['hilo_diff'].iloc[index] < 0, data['hilo_diff'].iloc[index - 1] > 0):
                data['day_buffer'].iloc[index] = -1
                data['hilo_diff_buffer'].iloc[index] = data['hilo_diff'].iloc[index]
                data_xy = data_xy.append(data.iloc[index-1], ignore_index=True)
            elif np.logical_and(data['hilo_diff'].iloc[index] < 0, data['hilo_diff'].iloc[index - 1] < 0):
                data['day_buffer'].iloc[index] = data['day_buffer'].iloc[index - 1] - 1
                data['hilo_diff_buffer'].iloc[index] = data['hilo_diff_buffer'].iloc[index - 1] + data['hilo_diff'].iloc[index]

        data_xy = data_xy.astype(data.dtypes)
    
        output = data_xy
        
    return(output)

def AR_prediction(input_data, ar_lags = 1, split_train_ratio = 0.8, name = ''):

    results_df = pd.DataFrame(columns=['model', 'mean', 'stdev', '68p', '95p', '99p', 'autocorelation_dw', 'homoscedasticity', 'normality', 'akaike'])

    adfuller(input_data)
    #second value is P-value. Series is strationary >5%

    #testing Autokorelation
    plot_acf(input_data, alpha =0.05)
    plot_pacf(input_data, alpha =0.05, lags=4)
    #seems like AR(1) 

    # Strengh of Lag 1 autocorrelation
    dataframe = concat([input_data.shift(1), input_data], axis=1)
    dataframe.columns = ['t-1', 't+1']
    dataframe.corr()

    #Traintest split
    train_size = int(round(-len(input_data)*split_train_ratio,0))
    train_df = input_data[:-train_size]
    test_df = input_data[-train_size:]
    #nasdaq_train_df = data[:-train_size]
    #nasdaq_test_df = data[-train_size:]

    #Model estimation
    model = AutoReg(train_df, lags=ar_lags)
    model_fit = model.fit()
    print('Coefficients: %s' % model_fit.params)

    #Prediction
    predictions = model_fit.predict(start=len(train_df), end=len(train_df)+len(test_df)-1, dynamic=False)
    rmse = sqrt(mean_squared_error(test_df, predictions))
    print('Test RMSE: %.3f' % rmse)

    # Diagnostics

    residuals = predictions - test_df

    # homoscedasticity 

    homoscedasticity = sms.het_arch(residuals)[3]

    # autocorelation
    plot_acf(residuals, alpha =0.05)
    plot_pacf(residuals, alpha =0.05, lags=4)
    plt.show()

    autocorelation = durbin_watson(residuals)

    # normality 

    normality = stats.jarque_bera(residuals)[1]

    #Amount of outliers in entire dataset according to prediction
    outliers = input_data[abs(input_data) > abs(0.000876 + 3*stdev(input_data))]
    len(outliers)/len(input_data)

    #akaike information criterion

    akaike = model_fit.aic

    # Plotting prediction

    plt.plot(predictions)
    plt.plot(predictions + 3*stdev(train_df))
    plt.plot(predictions - 3*stdev(train_df))
    plt.show()

    # Plotting entire series
    plt.plot(predictions)
    plt.plot(predictions + 3*stdev(train_df))
    plt.plot(predictions - 3*stdev(train_df))
    plt.plot(test_df)
    plt.plot(train_df)
    plt.show()

    # recording results

    results_dict = {
        'model': name,
        'mean': np.mean(predictions),
        'stdev': stdev(train_df),
        'autocorelation_dw': autocorelation,
        'homoscedasticity': homoscedasticity,
        'normality': normality,
        'akaike': akaike
    }

    results_df = results_df.append(results_dict, ignore_index=True)

    if np.mean(input_data) > 0:

        bounds = ['68p', '95p', '99p']
        for i in range(3):
            results_df[bounds[i]] = results_df['mean'] + (i+1) * results_df['stdev']
    else:
        bounds = ['68p', '95p', '99p']
        for i in range(3):
            results_df[bounds[i]] = results_df['mean'] - (i+1) * results_df['stdev']

    return(results_df)

# %%
# Parameters

# Data
ticker = '^IXIC'
start_date = '2000-12-1'
end_date = '2020-12-31'
yh_start_date = int(time.mktime(datetime.datetime(2000, 12, 1, 23, 59).timetuple()))
yh_end_date = int(time.mktime(datetime.datetime(2020, 12, 31, 23, 59).timetuple()))
interval = '1d'

# Model
split_train_ratio = 0.8
AR_lags = 1

# %%
# Loading data from yahoo

query_string1 = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={yh_start_date}&period2={yh_end_date}&interval={interval}&events=history&includeAdjustedClose=true'

nasdaq_df = pd.read_csv(query_string1)

results_df = pd.DataFrame(columns=['model', 'mean', 'stdev', '68p', '95p', '99p'])

# %%
# Data preparation

data = yields_preprocessing(nasdaq_df)
data_xy_analysis = yields_preprocessing(nasdaq_df, clustering=True)

# %%
# Daily Yields

results_df = AR_prediction(data['hilo_diff'], 
                                ar_lags=AR_lags, 
                                split_train_ratio=split_train_ratio, 
                                name='AR1 daily yields')

# Cluestered Yields
results_df = results_df.append(AR_prediction(data_xy_analysis['hilo_diff_buffer'], 
                                ar_lags=AR_lags, 
                                split_train_ratio=split_train_ratio, 
                                name='AR1 clustered yields'), ignore_index=True)

# Cluestered days
results_df = results_df.append(AR_prediction(data_xy_analysis['day_buffer'], 
                                ar_lags=AR_lags, 
                                split_train_ratio=split_train_ratio, 
                                name='AR1 clustered days'), ignore_index=True)

# Positive Yields only
results_df = results_df.append(AR_prediction(data_xy_analysis['hilo_diff_buffer'][data_xy_analysis['hilo_diff_buffer']>0].reset_index(drop=True), 
                                ar_lags=2, 
                                split_train_ratio=split_train_ratio, 
                                name='AR1 clustered positive yields'), ignore_index=True)

# Negative Yields only
results_df = results_df.append(AR_prediction(data_xy_analysis['hilo_diff_buffer'][data_xy_analysis['hilo_diff_buffer']<0].reset_index(drop=True), 
                                ar_lags=2, 
                                split_train_ratio=split_train_ratio, 
                                name='AR1 clustered negative yields'), ignore_index=True)

# Positive days only
results_df = results_df.append(AR_prediction(data_xy_analysis['day_buffer'][data_xy_analysis['hilo_diff_buffer']>0].reset_index(drop=True), 
                                ar_lags=AR_lags, 
                                split_train_ratio=split_train_ratio, 
                                name='AR1 clustered positive days'), ignore_index=True)

# Negative days only
results_df = results_df.append(AR_prediction(data_xy_analysis['day_buffer'][data_xy_analysis['hilo_diff_buffer']<0].reset_index(drop=True), 
                                ar_lags=AR_lags, 
                                split_train_ratio=split_train_ratio, 
                                name='AR1 clustered negative days'), ignore_index=True)

# %%