import pandas as pd
from base_learners import autoarima_learner, ets_learner, theta_learner, absolute_error
import xgboost as xg
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

all_data_ts = pd.read_csv('src/Features/all_data_ts.csv', sep=',')
stores_list = all_data_ts['unique_id'].unique().tolist()

def xgboost_learner(df):
    ensemble_model = xg.Booster()
    ensemble_model.load_model('src/model/booster_model.bin')
    df_ts = df.copy()
    df_ts = df_ts[['ds', 'y']]
    df_ts = df_ts.set_index('ds')
    ensemble_pred = ensemble_model.predict(xg.DMatrix(df_ts))
    return ensemble_pred


# ensemble_preds = []
mapes = pd.DataFrame()
for s in stores_list:
    print(s)
    df_predictions = pd.DataFrame()
    new_row_df = pd.DataFrame()
    additional_predictions = pd.DataFrame()
    df_ts = all_data_ts[all_data_ts['unique_id']==s]
    arima_pred = autoarima_learner(df_ts)
    ets_pred = ets_learner(df_ts)
    theta_pred = theta_learner(df_ts)
    ensemble_pred = xgboost_learner(df_ts)
    # ensemble_preds.append(ensemble_pred)

    error_Arima = (ensemble_pred[:, 0].mean())
    error_ETS = (ensemble_pred[:, 1].mean())
    error_Theta = (ensemble_pred[:, 2].mean())
    weight_Arima = 1 / error_Arima
    weight_ETS = 1 / error_ETS
    weight_Theta = 1 / error_Theta
    total_weight = weight_Arima + weight_ETS + weight_Theta

    normalized_weight_Arima =  weight_Arima / total_weight
    normalized_weight_ETS = weight_ETS / total_weight
    normalized_weight_Theta = weight_Theta / total_weight
    print('W Arima:', normalized_weight_Arima)
    print('W ETS:', normalized_weight_ETS)
    print('W Theta:', normalized_weight_Theta)
    print('----------------------------------')


    # Convert 'ds' column in df_ts to datetime and format it
    df_ts['ds'] = pd.to_datetime(df_ts['ds'])
    df_ts['ds'] = df_ts['ds'].dt.strftime('%Y-%m-%d')

    # Extract the last row from df_ts
    last_row_ts = df_ts.iloc[-25]

    # Create a new row for df_predictions using the values from the last row of df_ts
    new_row = {
        'Store': s,
        'ds': last_row_ts['ds'],
        'y': last_row_ts['y'],
        'Arima': last_row_ts['y'],  # Replace with your specific value
        'ETS': last_row_ts['y'],   # Replace with your specific value
        'Theta': last_row_ts['y'],  # Replace with your specific value
        'Metalearner': last_row_ts['y']  # Replace with your specific value
    }

    # Convert the dictionary to a DataFrame
    new_row_df = pd.DataFrame([new_row])
    df_predictions = pd.concat([df_predictions, new_row_df], axis=0)

    # Extract the last 23 rows from df_ts
    last_23_rows = df_ts.iloc[-24:]

    additional_predictions['ds'] = pd.to_datetime(last_23_rows['ds']).dt.strftime('%Y-%m-%d')
    additional_predictions['y'] = last_23_rows['y']
    additional_predictions['Arima'] =  arima_pred.iloc[:, 1].values
    additional_predictions['ETS'] = ets_pred['simulation'].values
    additional_predictions['Theta'] = theta_pred['forecast'].values

    additional_predictions['Metalearner'] = (
        additional_predictions['Arima'].values * normalized_weight_Arima +
        additional_predictions['ETS'].values * normalized_weight_ETS +
        additional_predictions['Theta'].values * normalized_weight_Theta
    )
   
    df_predictions = pd.concat([df_predictions.iloc[:2], additional_predictions], ignore_index=True, axis=0)

    df_predictions['Store'] = s  # Replace with your specific value

    mape_ensemble = absolute_error(df_predictions['y'], df_predictions['Metalearner'])
    mape_arima = absolute_error(df_predictions['y'], df_predictions['Arima'])
    mape_ets = absolute_error(df_predictions['y'], df_predictions['ETS'])
    mape_theta = absolute_error(df_predictions['y'], df_predictions['Theta'])

    mape_df = pd.DataFrame({
        'Store':[s],
        'Arima': [mape_arima],
        'ETS': [mape_ets],
        'Theta': [mape_theta],
        'Metalearner': [mape_ensemble]
                            })
    mapes = pd.concat([mapes, mape_df], axis=0, ignore_index=True)
    print(mape_df)


    # df_ts = df_ts.iloc[:-24]
   
    # plt.figure(figsize=(10, 6))

    # # Plotting the original time series
    # plt.plot(df_ts['ds'].tail(50), df_ts['y'].tail(50), label='Original',linewidth=1.5, linestyle='-', color='black')

    # # Plotting the predictions
    # plt.plot(df_predictions['ds'], df_predictions['Arima'], label='Arima',linewidth=1, linestyle='-.')
    # plt.plot(df_predictions['ds'], df_predictions['ETS'], label='ETS',linewidth=1, linestyle='-.')
    # plt.plot(df_predictions['ds'], df_predictions['Theta'], label='Theta',linewidth=1, linestyle='-.')
    # plt.plot(df_predictions['ds'], df_predictions['Metalearner'], label='Metalearner',linewidth=1, linestyle='-.')

    # plt.title(f'Time Series for Store {s}')
    # plt.xlabel('Date')
    # plt.ylabel('Value')
    # plt.legend()

    # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # # plt.clf()
    
    # plt.savefig(f'./img/forecasting/plot_{s}.png')
    # plt.show(block=False)
    # plt.pause(0.1)

# plt.show()
print(mapes)
mapes.to_csv('src/mapes/mapes.csv', sep=',')

    



    # print(ensemble_pred)

    

