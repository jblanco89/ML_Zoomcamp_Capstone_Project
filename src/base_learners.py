"""
Time Series Forecasting Script

This script performs time series forecasting using three different models as base learners:

        AutoARIMA, 
        ETS (Exponential Smoothing State Space Model), and 
        Theta.

It loads a time series dataset, splits it into train, validation, and test sets, 
and then applies the forecasting models to estimate targets for each store. 
The results are saved in a CSV file, and feature data for training, validation, and testing are 
also saved separately.

Author: Javier Blanco
Date: Dic-10-2023
"""

import pandas as pd
from tsfeatures import tsfeatures 
import pmdarima as pm
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.forecasting.theta import ThetaModel

# Upload time series dataset and then we need to splitted in train, validation and test data
# proportion could be 80:10:10

def split_train_test_data(df:pd.DataFrame, s:str='store_1', train_prop:float = 0.8)->tuple:
    d = df[df['unique_id'] == s]
    d_new = d[['unique_id','ds', 'y']]
    n_train = train_prop
    n_val = 0.1
    n_test = 0.1
    train = d_new[:int(d_new.shape[0]*n_train)]
    val = d_new[int(d_new.shape[0]*n_train):int(d_new.shape[0]*n_val) + int(d_new.shape[0]*n_train)]
    test = d_new[:int(d_new.shape[0]*n_test)]

    return train, val, test

def features_by_dataset(df:pd.DataFrame)->tuple:
    df_ts = df.copy()
    val_df = pd.DataFrame()
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    stores_list = df_ts['unique_id'].unique().tolist()
    for s in stores_list:
        train, val, test = split_train_test_data(df_ts, train_prop=0.8, s=s)
        val_df = pd.concat([val_df, val], axis=0)
        train_df = pd.concat([train_df, train], axis=0)
        test_df = pd.concat([test_df, test], axis=0)
    # print(val_df)
    val_df['ds'] = pd.to_datetime(val_df['ds'])
    train_df['ds'] = pd.to_datetime(train_df['ds'])
    test_df['ds'] = pd.to_datetime(test_df['ds'])

    train_features = tsfeatures(train_df, freq=1)
    val_features = tsfeatures(val_df, freq=1)
    test_features = tsfeatures(test_df, freq=1)

    train_features = train_features.fillna(0)
    test_features = test_features.fillna(0)
    val_features = val_features.fillna(0)

    return train_features, val_features, test_features


def autoarima_learner(df:pd.DataFrame)-> pd.DataFrame:
    df_ts = df.copy()
    df_ts = df_ts[['ds', 'y']]
    df_ts = df_ts.set_index('ds')
    history = pm.auto_arima(df_ts, start_p=0, start_q=0, seasonal=True, stationary=True, max_p=5, m=1)
    model_summary = history.summary()
    predictions = history.fit_predict(df_ts, n_periods=24)
    predictions = predictions.reset_index()
    # print(model_summary)

    return predictions

def ets_learner(df:pd.DataFrame)->pd.DataFrame:
    df_ts = df.copy()
    df_ts = df_ts[['ds', 'y']]
    df_ts = df_ts.set_index('ds')
    history = ETSModel(df_ts['y'], error='add', trend='add', seasonal_periods=4, seasonal='add')
    results = history.fit()
    predictions = results.forecast(steps=24)
    predictions = predictions.reset_index()

    return predictions

def theta_learner(df:pd.DataFrame)-> pd.DataFrame:
    df_ts = df.copy()
    df_ts = df_ts[['ds', 'y']]
    df_ts = df_ts.set_index('ds')
    history = ThetaModel(endog=df_ts['y'], deseasonalize=True, period=4)
    results = history.fit()
    predictions = results.forecast(steps=24)
    predictions = predictions.reset_index()
    return predictions


def absolute_error(y_val:pd.DataFrame, y_pred:pd.DataFrame)->float:
    mape = mean_absolute_percentage_error(y_val, y_pred)
    return mape

def base_learners_predictions(df, store)->tuple:
    df_ts = df.copy()
    df_ts = df_ts[df_ts['unique_id']==store]
    y_arima = autoarima_learner(df_ts)
    y_ets = ets_learner(df_ts)
    y_theta = theta_learner(df_ts)
    y_train, y_val, y_test = split_train_test_data(df,s=store, train_prop=0.8)

    mape_arima = absolute_error(y_val=y_val['y'], y_pred=y_arima.iloc[:,1])
    mape_ets = absolute_error(y_val=y_val['y'], y_pred=y_ets['simulation'])
    mape_theta = absolute_error(y_val=y_val['y'], y_pred=y_theta['forecast'])

    return mape_arima, mape_ets, mape_theta


def estimate_targets(df:pd.DataFrame, stores:list)->pd.DataFrame:
    stores_list = stores
    df_ts = df.copy()
    loss_arima_list = []
    loss_ets_list = []
    loss_theta_list = []
    for s in stores_list:
        loss_arima, loss_ets, loss_theta = base_learners_predictions(df_ts, store=s)
        loss_arima_list.append(loss_arima)
        loss_ets_list.append(loss_ets)
        loss_theta_list.append(loss_theta)
    losses_df = pd.DataFrame({
    'y_arima': loss_arima_list,
    'y_ets': loss_ets_list,
    'y_theta': loss_theta_list
    })
    losses_df.to_csv('src/Targets/targets.csv', sep=',')
    print(losses_df)
    return losses_df

if __name__ == '__main__':
    all_ts = pd.read_csv('src/Features/all_data_ts.csv', sep=',')
    stores_list = all_ts['unique_id'].unique().tolist()
    targets = estimate_targets(all_ts, stores=stores_list)
    X_train, X_val, X_test = features_by_dataset(all_ts)
    X_train.to_csv('src/Features/X_train.csv', sep=',')
    X_test.to_csv('src/Features/X_test.csv', sep=',')
    X_val.to_csv('src/Features/X_val.csv', sep=',')

