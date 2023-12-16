import pandas as pd
import xgboost as xg
from base_learners import absolute_error

# best_params:
# param_grid = {
#     'n_estimators': 10,
#     'max_depth': 10,
#     'eta': 0.1,
#     'subsample':0.90,
#     'colsample_bytree': 0.77
#         }

X = pd.read_csv('src/Features/X_train.csv', sep=',')
y = pd.read_csv('src/Targets/targets.csv', sep=',')

X = X.drop(columns=['Unnamed: 0', 'unique_id'])
y = y.drop(columns='Unnamed: 0')

X_test = pd.read_csv('src/Features/X_test.csv', sep=',').drop(columns=['Unnamed: 0', 'unique_id'])

model = xg.XGBRegressor(eta=0.1,
    n_estimators=10,
    max_depth=10,
    subsample=0.9,
    colsample_bytree=0.77,
    objective='reg:absoluteerror',
    eval_metric='mape')

history = model.fit(X, y)
print(history)

y_pred = history.predict(X_test)


mape = absolute_error(y_val=y, y_pred=y_pred)
print(mape)

model.save_model('src/model/booster_model.bin')