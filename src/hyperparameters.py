import pandas as pd
import xgboost as xg
from base_learners import absolute_error
import mlflow
import time

def main(X_train, X_val, y_train, y_val):
    param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [10, 14, 20],
    'eta': [0.1, 0.6, 1.0],
    'subsample': [0.8, 0.90, 1.0],
    'colsample_bytree': [0.77, 0.8, 0.9],
        }
    for n_estimators in param_grid['n_estimators']:
        for max_depth in param_grid['max_depth']:
            for eta in param_grid['eta']:
                for subsample in param_grid['subsample']:
                    for colsample_bytree in param_grid['colsample_bytree']:
                        xg_model_metalearner = xg.XGBRegressor(
                            objective='reg:absoluteerror',
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            eta=eta,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            eval_metric='mape',
                            random_state=123
                            )

                        xg_model_metalearner.fit(X_train, y_train)

                        y_pred = xg_model_metalearner.predict(X_val)

                        mape = absolute_error(y_val, y_pred)

                        l = [n_estimators,max_depth, eta, subsample, colsample_bytree, mape]
                        with mlflow.start_run(nested=True):
                            mlflow.log_metric('mape', l[5])
                            mlflow.log_param('n_estimators', l[0])
                            mlflow.log_param('max_depth', l[1])
                            mlflow.log_param('eta', l[2])
                            mlflow.log_param('subsample', l[3])
                            mlflow.log_param('colsample_bytree', l[4])
                        mlflow.end_run()
                        time.sleep(2)
                        print(l)

    
    


if __name__ == '__main__':
    X = pd.read_csv('src/Features/X_train.csv', sep=',')
    y = pd.read_csv('src/Targets/targets.csv', sep=',')

    X = X.drop(columns=['Unnamed: 0', 'unique_id'])
    y = y.drop(columns='Unnamed: 0')
    X_test = pd.read_csv('src/Features/X_test.csv', sep=',').drop(columns=['Unnamed: 0', 'unique_id'])
    # print(y)

    
    mlflow.set_tracking_uri('http://127.0.0.1:5000/')
    mlflow.set_experiment('ML Zoomcamp Capstone Project')
    mlflow.set_tag("author", "Javier Blanco")
    mlflow.set_tag("description", "Tunning Hyperparameter of XgBoost Ensemble")
    mlflow.end_run()
    with mlflow.start_run():
        main(X_train=X, X_val=X_test, y_train=y, y_val=y)



