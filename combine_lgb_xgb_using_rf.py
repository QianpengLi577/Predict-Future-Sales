import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings(action='ignore')


y_train = pd.read_csv('y_train.csv')
y_valid = pd.read_csv('y_valid.csv')

y_train_lgb = pd.read_csv('y_train_lgb.csv',header=None)
y_train_xgb = pd.read_csv('y_train_xgb.csv',header=None)

y_test_lgb = pd.read_csv('y_test_lgb.csv',header=None)
y_test_xgb = pd.read_csv('y_test_xgb.csv',header=None)

y_valid_lgb = pd.read_csv('y_valid_lgb.csv',header=None)
y_valid_xgb = pd.read_csv('y_valid_xgb.csv',header=None)

combine_test = pd.DataFrame(y_test_lgb.values, columns=['lgb'])
combine_test['xgb']=y_test_xgb.values

combine_train = pd.DataFrame(y_train_lgb.values, columns=['lgb'])
combine_train['xgb']=y_train_xgb.values

combine_val = pd.DataFrame(y_valid_lgb.values, columns=['lgb'])
combine_val['xgb']=y_valid_xgb.values

from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0, n_jobs=-1)
rf_model.fit(combine_train, y_train)

y_train_rf = rf_model.predict(combine_train)
y_valid_rf = rf_model.predict(combine_val)
y_test_rf = rf_model.predict(combine_test)

from sklearn.metrics import mean_squared_error
print('Train rmse:', np.sqrt(mean_squared_error(y_train, y_train_rf)))
print('Validation rmse:', np.sqrt(mean_squared_error(y_valid, y_valid_rf)))
np.savetxt('y_train_rf.csv',y_train_rf,delimiter=',',encoding='utf-8-sig',fmt='%.12f')
np.savetxt('y_valid_rf.csv',y_valid_rf,delimiter=',',encoding='utf-8-sig',fmt='%.12f')
np.savetxt('y_test_rf.csv',y_test_rf,delimiter=',',encoding='utf-8-sig',fmt='%.12f')

submission = pd.read_csv('sample_submission.csv')
submission['item_cnt_month'] = y_test_rf
submission.to_csv('submission_com_rf.csv', index=False)