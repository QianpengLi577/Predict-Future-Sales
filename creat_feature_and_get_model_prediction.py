# %%

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings(action='ignore')

# 加载数据
test = pd.read_csv('test.csv', dtype={'ID': 'int32', 'shop_id': 'int32',
                                      'item_id': 'int32'})
item_categories = pd.read_csv('item_categories.csv',
                              dtype={'item_category_name': 'str', 'item_category_id': 'int32'})
items = pd.read_csv('items.csv', dtype={'item_name': 'str', 'item_id': 'int32',
                                        'item_category_id': 'int32'})
shops = pd.read_csv('shops.csv', dtype={'shop_name': 'str', 'shop_id': 'int32'})
sales_train = pd.read_csv('sales_train.csv', parse_dates=['date'],
                          dtype={'date': 'str', 'date_block_num': 'int32', 'shop_id': 'int32',
                                 'item_id': 'int32', 'item_price': 'float32', 'item_cnt_day': 'int32'})

# %%

# 对训练的数据进行筛选
sales_train = sales_train.query('item_cnt_day >=0 and item_cnt_day <= 1000')
sales_train = sales_train.query('item_price >=0 and item_price <= 50000')

# %%

# 获得每个商店每个date的count
sales_by_shop_id = sales_train.pivot_table(index=['shop_id'], values=['item_cnt_day'],
                                           columns='date_block_num', aggfunc=np.sum, fill_value=0).reset_index()
sales_by_shop_id.columns = sales_by_shop_id.columns.droplevel().map(str)
sales_by_shop_id = sales_by_shop_id.reset_index(drop=True).rename_axis(None, axis=1)
sales_by_shop_id.columns.values[0] = 'shop_id'
print('sales_by_shop_id:')
print(sales_by_shop_id)
print('经过分析，可以发发现商店id为：', [0, 1, 8, 11, 13, 17, 23, 27, 29, 30, 32, 33, 40, 43, 54], '可能已经关闭了')
print('商店id为：', [9, 20], '为一年一开，并且当为第34个月时，处于关闭状态')
print('商店id为:', [36], '为新开的店铺')
print('通过分析商店的id与name，可以发现0-57是继承关系，1-58是继承关系，11-10也是继承关系，而40-39可能是分店暂时开业关系')

# %%

# 根据上面的分析，对训练集商店id替换
sales_train.loc[sales_train['shop_id'] == 0, 'shop_id'] = 57
sales_train.loc[sales_train['shop_id'] == 1, 'shop_id'] = 58
sales_train.loc[sales_train['shop_id'] == 11, 'shop_id'] = 10
sales_train.loc[sales_train['shop_id'] == 40, 'shop_id'] = 39
# 合并测试集的店铺
test.loc[test['shop_id'] == 0, 'shop_id'] = 57
test.loc[test['shop_id'] == 1, 'shop_id'] = 58
test.loc[test['shop_id'] == 11, 'shop_id'] = 10
test.loc[test['shop_id'] == 40, 'shop_id'] = 39

# %%

# 获得测试集用到的shopid并去除不在测试集的shop
test_shop_ids = test['shop_id'].unique()
sales_train = sales_train[sales_train['shop_id'].isin(test_shop_ids)]

# %%

# 输出shop的name
print(shops['shop_name'])
shops['city'] = shops['shop_name'].apply(lambda x: x.split()[0])
shops['city'].unique()
shops.loc[shops['city'] == '!Якутск', 'city'] = 'Якутск'
# 获得shopname
print(shops['city'])

# %%

# 对商店的城市进行编码
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
shops['city'] = label_encoder.fit_transform(shops['city'])
shops = shops.drop('shop_name', axis=1)
print(shops['city'])

# %%

print('商品的类别：')
print(item_categories['item_category_name'])
# 第一个字符串为大的类别，获得其类别，对其进行编码
item_categories['category'] = item_categories['item_category_name'].apply(lambda x: x.split()[0])


def elsecategory(x):
    if len(item_categories[item_categories['category'] == x]) >= 6:
        return x
    else:
        return 'else'


item_categories['category'] = item_categories['category'].apply(elsecategory)
# 对类别进行编码
label_encoder = LabelEncoder()
item_categories['category'] = label_encoder.fit_transform(item_categories['category'])
item_categories = item_categories.drop('item_category_name', axis=1)

# %%

# 获得商品第一次售出的data block num，并将未售出的设置为测试集的月份
items = items.drop(['item_name'], axis=1)
items['first_sale_date'] = sales_train.groupby('item_id').agg({'date_block_num': 'min'})['date_block_num']
items['first_sale_date'] = items['first_sale_date'].fillna(34)

# %%

# 获得date shop item的数据集
from itertools import product

train = []
for i in sales_train['date_block_num'].unique():
    all_shop = sales_train.loc[sales_train['date_block_num'] == i, 'shop_id'].unique()
    all_item = sales_train.loc[sales_train['date_block_num'] == i, 'item_id'].unique()
    train.append(np.array(list(product([i], all_shop, all_item))))
train = pd.DataFrame(np.vstack(train), columns=['date_block_num', 'shop_id', 'item_id'])
print(train.head(n=10))

# %%

import gc

# 计算 销售shop item date 销售总个数(月)、销售平均每次多少个、每个月几次、单价均值
group = sales_train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day': ['sum', 'count'],
                                                                           'item_price': 'mean'})
group = group.reset_index()
group.columns = ['date_block_num', 'shop_id', 'item_id', 'item_cnt_month',
                 'item_count', 'item_price_mean']
train = train.merge(group, on=['date_block_num', 'shop_id', 'item_id'], how='left')
# train['item_cnt_month'] = train['item_cnt_month'].clip(0, 20)
print('train.head:')
print(train.head())

# 内存占用太大了。。。。。。giao
del group
gc.collect();

# %%

# 对测试集合并
test['date_block_num'] = 34
all_data = pd.concat([train, test.drop('ID', axis=1)],
                     ignore_index=True,
                     keys=['date_block_num', 'shop_id', 'item_id'])
all_data = all_data.fillna(0)
print('all_data.head():')
print(all_data.head())

# %%

# 增加 city id、类别id、大类id、第一次售出date
all_data = all_data.merge(shops, on='shop_id', how='left')
all_data = all_data.merge(items, on='item_id', how='left')
all_data = all_data.merge(item_categories, on='item_category_id', how='left')

# %%

# 增加每个date item的均值
group = all_data.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': 'mean'})
group = group.reset_index()
group = group.rename(columns={'item_cnt_month': 'date_item_avg'})
all_data = all_data.merge(group, on=['date_block_num', 'item_id'], how='left')
# 清内存
del group
gc.collect();

# %%

# 增加date item city -mean
group = all_data.groupby(['date_block_num', 'item_id', 'city']).agg({'item_cnt_month': 'mean'})
group = group.reset_index()
group = group.rename(columns={'item_cnt_month': 'date_item_city_avg'})
all_data = all_data.merge(group, on=['date_block_num', 'item_id', 'city'], how='left')
# 清内存
del group
gc.collect();

# %%

# 增加date shop category_id  -mean
group = all_data.groupby(['date_block_num', 'shop_id', 'item_category_id']).agg({'item_cnt_month': 'mean'})
group = group.reset_index()
group = group.rename(columns={'item_cnt_month': 'date_shop_categoryid_avg'})
all_data = all_data.merge(group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
# 清内存
del group
gc.collect();

# 增加date shop category -mean
group = all_data.groupby(['date_block_num', 'shop_id', 'category']).agg({'item_cnt_month': 'mean'})
group = group.reset_index()
group = group.rename(columns={'item_cnt_month': 'date_shop_category_avg'})
all_data = all_data.merge(group, on=['date_block_num', 'shop_id', 'category'], how='left')
# 清内存
del group
gc.collect();

# %%

# 增加date shop item的cnt三个月的延时
tmp = all_data[['date_block_num', 'shop_id', 'item_id', 'item_cnt_month']].copy()
for i in [1, 2, 3]:
    group = tmp.copy()
    group = group.rename(columns={'item_cnt_month': 'item_cnt_month' + '_delay_' + str(i)})
    group['date_block_num'] += i
    all_data = all_data.merge(group.drop_duplicates(), on=['date_block_num', 'shop_id', 'item_id'], how='left')
    all_data['item_cnt_month' + '_delay_' + str(i)] = all_data['item_cnt_month' + '_delay_' + str(i)].clip(0, 20)
    print('adding ' + 'item_cnt_month' + '_delay_' + str(i))
    del group
    gc.collect()
del tmp
gc.collect();

# %%

# 增加date shop item的count三个月的延时
tmp = all_data[['date_block_num', 'shop_id', 'item_id', 'item_count']].copy()
for i in [1, 2, 3]:
    print('adding ' + 'item_count' + '_delay_' + str(i))
    group = tmp.copy()
    group = group.rename(columns={'item_count': 'item_count' + '_delay_' + str(i)})
    group['date_block_num'] += i
    all_data = all_data.merge(group.drop_duplicates(), on=['date_block_num', 'shop_id', 'item_id'], how='left')
    all_data['item_count' + '_delay_' + str(i)] = all_data['item_count' + '_delay_' + str(i)].clip(0, 20)
    del group
    gc.collect()
del tmp
gc.collect();

# %%

# 增加date shop item的price mean三个月的延时
tmp = all_data[['date_block_num', 'shop_id', 'item_id', 'item_price_mean']].copy()
for i in [1, 2, 3]:
    print('adding ' + 'item_price_mean' + '_delay_' + str(i))
    group = tmp.copy()
    group = group.rename(columns={'item_price_mean': 'item_price_mean' + '_delay_' + str(i)})
    group['date_block_num'] += i
    all_data = all_data.merge(group.drop_duplicates(), on=['date_block_num', 'shop_id', 'item_id'], how='left')
    all_data['item_price_mean' + '_delay_' + str(i)] = all_data['item_price_mean' + '_delay_' + str(i)].clip(0, 20)
    del group
    gc.collect()
del tmp
gc.collect();

# %%

# 增加date  item的cnt mean三个月的延时
tmp = all_data[['date_block_num', 'item_id', 'date_item_avg']].copy()
for i in [1, 2, 3]:
    print('adding ' + 'date_item_avg' + '_delay_' + str(i))
    group = tmp.copy()
    group = group.rename(columns={'date_item_avg': 'date_item_avg' + '_delay_' + str(i)})
    group['date_block_num'] += i
    all_data = all_data.merge(group.drop_duplicates(), on=['date_block_num', 'item_id'], how='left')
    all_data['date_item_avg' + '_delay_' + str(i)] = all_data['date_item_avg' + '_delay_' + str(i)].clip(0, 20)
    del group
    gc.collect()
del tmp
gc.collect();
all_data = all_data.drop('date_item_avg', axis=1)
# %%

# 增加date  item city 的cnt mean三个月的延时
tmp = all_data[['date_block_num', 'shop_id', 'item_id', 'date_item_city_avg']].copy()
for i in [1, 2, 3]:
    print('adding ' + 'date_item_city_avg' + '_delay_' + str(i))
    group = tmp.copy()
    group = group.rename(columns={'date_item_city_avg': 'date_item_city_avg' + '_delay_' + str(i)})
    group['date_block_num'] += i
    all_data = all_data.merge(group.drop_duplicates(), on=['date_block_num', 'shop_id', 'item_id'], how='left')
    all_data['date_item_city_avg' + '_delay_' + str(i)] = all_data['date_item_city_avg' + '_delay_' + str(i)].clip(0,
                                                                                                                   20)
    del group
    gc.collect()
del tmp
gc.collect();
all_data = all_data.drop('date_item_city_avg', axis=1)

# %%

# 增加date  shop cageid 的cnt mean三个月的延时
tmp = all_data[['date_block_num', 'shop_id', 'item_category_id', 'date_shop_categoryid_avg']].copy()
for i in [1, 2, 3]:
    print('adding ' + 'date_shop_categoryid_avg' + '_delay_' + str(i))
    group = tmp.copy()
    group = group.rename(columns={'date_shop_categoryid_avg': 'date_shop_categoryid_avg' + '_delay_' + str(i)})
    group['date_block_num'] += i
    all_data = all_data.merge(group.drop_duplicates(), on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
    all_data['date_shop_categoryid_avg' + '_delay_' + str(i)] = all_data[
        'date_shop_categoryid_avg' + '_delay_' + str(i)].clip(0, 20)
    del group
    gc.collect()
del tmp
gc.collect();
all_data = all_data.drop('date_shop_categoryid_avg', axis=1)


#%%

# 增加date  shop cage 的cnt mean三个月的延时
tmp = all_data[['date_block_num', 'shop_id', 'category', 'date_shop_category_avg']].copy()
for i in [1, 2, 3]:
    print('adding ' + 'date_shop_category_avg' + '_delay_' + str(i))
    group = tmp.copy()
    group = group.rename(columns={'date_shop_category_avg': 'date_shop_category_avg' + '_delay_' + str(i)})
    group['date_block_num'] += i
    all_data = all_data.merge(group.drop_duplicates(), on=['date_block_num', 'shop_id', 'category'], how='left')
    all_data['date_shop_category_avg' + '_delay_' + str(i)] = all_data[
        'date_shop_category_avg' + '_delay_' + str(i)].clip(0, 20)
    del group
    gc.collect()
del tmp
gc.collect();
all_data = all_data.drop('date_shop_category_avg', axis=1)

#%%

for i in [1, 2, 3]:
    print('adding ' + 'item_price_mean' + '_tend_' + str(i))
    if i == 1:
        group1 = all_data[['date_block_num', 'shop_id', 'item_id', 'item_price_mean' + '_delay_' + str(i)]].copy()
        group1['item_price_mean' + '_delay_' + str(i)] = (group1['item_price_mean' + '_delay_' + str(i)] - all_data[
            'item_price_mean']) / all_data['item_price_mean']
        group1 = group1.rename(columns={'item_price_mean' + '_delay_' + str(i): 'item_price_mean' + '_tend_' + str(i)})
    else:
        group1 = all_data[['date_block_num', 'shop_id', 'item_id', 'item_price_mean' + '_delay_' + str(i)]].copy()
        group1['item_price_mean' + '_delay_' + str(i)] = (group1['item_price_mean' + '_delay_' + str(i)] - all_data[
            'item_price_mean' + '_delay_' + str(i - 1)]) / all_data['item_price_mean' + '_delay_' + str(i - 1)]
        group1 = group1.rename(columns={'item_price_mean' + '_delay_' + str(i): 'item_price_mean' + '_tend_' + str(i)})
    group1['date_block_num'] += i
    all_data = all_data.merge(group1.drop_duplicates(), on=['date_block_num', 'shop_id', 'item_id'], how='left')
    all_data['item_price_mean' + '_tend_' + str(i)] = all_data[
        'item_price_mean' + '_tend_' + str(i)].clip(0, 20)
    del group1
    gc.collect();



# %%

# 去除前三个月的信息
all_data = all_data.drop(all_data[all_data['date_block_num'] < 3].index)
# %%

# 添加延时均值
all_data['item_cnt_month_delay_mean'] = all_data[['item_cnt_month_delay_1',
                                                  'item_cnt_month_delay_2',
                                                  'item_cnt_month_delay_3']].mean(axis=1).clip(0, 20)
all_data['item_cnt_month'] = all_data['item_cnt_month'].clip(0, 20)
# %%

# 延时的grad
# all_data['delay_ratio1'] = (all_data['item_cnt_month_delay_1'])/all_data['item_cnt_month_delay_2']
# all_data['delay_ratio2'] = (all_data['item_cnt_month_delay_2'])/all_data['item_cnt_month_delay_3']
# %%

# 相对于上新过去了多少时间
# all_data['brand_new'] = all_data['first_sale_date'] == all_data['date_block_num']
all_data['duration_after_first_sale'] = all_data['date_block_num'] - all_data['first_sale_date']
# all_data = all_data.drop('first_sale_date', axis=1)
all_data = all_data.replace([np.inf, -np.inf], np.nan).fillna(0)
# all_data['item_cnt_month_delay_1','item_cnt_month_delay_2','item_cnt_month_delay_3'
#             ,'item_count_delay_1','item_count_delay_2','item_count_delay_3'
#             ,'item_price_mean_delay_1','item_price_mean_delay_2','item_price_mean_delay_3'
#             ,'date_item_avg_delay_1','date_item_avg_delay_2','date_item_avg_delay_3'
#             ,'date_item_city_avg_delay_1','date_item_city_avg_delay_2','date_item_city_avg_delay_3'
#             ,'date_shop_categoryid_avg_delay_1','date_shop_categoryid_avg_delay_2','date_shop_categoryid_avg_delay_3'
#             ,'item_cnt_month', 'item_cnt_month_lag_mean'] = all_data['item_cnt_month_delay_1','item_cnt_month_delay_2','item_cnt_month_delay_3'
#             ,'item_count_delay_1','item_count_delay_2','item_count_delay_3'
#             ,'item_price_mean_delay_1','item_price_mean_delay_2','item_price_mean_delay_3'
#             ,'date_item_avg_delay_1','date_item_avg_delay_2','date_item_avg_delay_3'
#             ,'date_item_city_avg_delay_1','date_item_city_avg_delay_2','date_item_city_avg_delay_3'
#             ,'date_shop_categoryid_avg_delay_1','date_shop_categoryid_avg_delay_2','date_shop_categoryid_avg_delay_3'
#             ,'item_cnt_month', 'item_cnt_month_lag_mean'].clip(0,20)
# %%

# 获得月份label
all_data['month'] = all_data['date_block_num'] % 12
all_data = all_data.drop(['item_price_mean', 'item_count'], axis=1)
all_data.info()

# all_data.to_csv('all_data.csv', index=False)

# %%

# 划分为训练集、验证集、测试集
X_train = all_data[all_data['date_block_num'] < 33]
X_train = X_train.drop(['item_cnt_month'], axis=1)
# Valid data (Features)
X_valid = all_data[all_data['date_block_num'] == 33]
X_valid = X_valid.drop(['item_cnt_month'], axis=1)
# Test data (Features)
X_test = all_data[all_data['date_block_num'] == 34]
X_test = X_test.drop(['item_cnt_month'], axis=1)

# Train data (Target values)
y_train = all_data[all_data['date_block_num'] < 33]['item_cnt_month']
# Valid data (Target values)
y_valid = all_data[all_data['date_block_num'] == 33]['item_cnt_month']


# X_train.to_csv('X_train.csv', index=False)
# X_valid.to_csv('X_valid.csv', index=False)
# X_test.to_csv('X_test.csv', index=False)
# y_train.to_csv('y_train.csv', index=False)
# y_valid.to_csv('y_valid.csv', index=False)
# Garbage collection
del all_data
gc.collect();

# %%

import lightgbm as lgb

# lightgbm的参数设置
# lgb hyper-parameters
params = {'metric': 'rmse',
          'num_leaves': 50,
          'learning_rate': 0.005,
          'feature_fraction': 0.75,
          'bagging_fraction': 0.75,
          'bagging_freq': 5,
          'force_col_wise': True,
          'random_state': 10,
          'num_boost_round': 2000}

cat_features = ['shop_id', 'city', 'item_category_id', 'category', 'month']

# lgb train and valid dataset
dtrain = lgb.Dataset(X_train, y_train)
dvalid = lgb.Dataset(X_valid, y_valid)

# Train LightGBM model
lgb_model = lgb.train(params=params,
                      train_set=dtrain,
                      num_boost_round=1500,
                      valid_sets=(dtrain, dvalid),
                      early_stopping_rounds=150,
                      categorical_feature=cat_features,
                      verbose_eval=100)

# %%

# 获得最终的预测


y_test_lgb = lgb_model.predict(X_test).clip(0, 20)
# np.savetxt('y_test_lgb.csv',y_test_lgb,delimiter=',',encoding='utf-8-sig',fmt='%.12f')
y_train_lgb = lgb_model.predict(X_train).clip(0, 20)
# np.savetxt('y_train_lgb.csv',y_train_lgb,delimiter=',',encoding='utf-8-sig',fmt='%.12f')
y_valid_lgb = lgb_model.predict(X_valid).clip(0, 20)
# np.savetxt('y_valid_lgb.csv',y_valid_lgb,delimiter=',',encoding='utf-8-sig',fmt='%.12f')

submission = pd.read_csv('sample_submission.csv')
submission['item_cnt_month'] = y_test_lgb
submission.to_csv('submission_lgb.csv', index=False)

#%%

#Train xgboost model

import xgboost as xgb

xgb_model = xgb.XGBRegressor(max_depth=8,
                             n_estimators=500,
                             min_child_weight=1000,
                             colsample_bytree=0.7,
                             subsample=0.7,
                             eta=0.3,
                             seed=0)
xgb_model.fit(X_train,
              y_train,
              eval_metric="rmse",
              eval_set=[(X_train, y_train), (X_valid, y_valid)],
              verbose=20,
              early_stopping_rounds=20)

y_train_xgb = xgb_model.predict(X_train)
y_valid_xgb = xgb_model.predict(X_valid)
y_test_xgb = xgb_model.predict(X_test)
# np.savetxt('y_train_xgb.csv',y_train_xgb,delimiter=',',encoding='utf-8-sig',fmt='%.12f')
# np.savetxt('y_valid_xgb.csv',y_valid_xgb,delimiter=',',encoding='utf-8-sig',fmt='%.12f')
# np.savetxt('y_test_xgb.csv',y_test_xgb,delimiter=',',encoding='utf-8-sig',fmt='%.12f')

submission = pd.read_csv('sample_submission.csv')
submission['item_cnt_month'] = y_test_xgb
submission.to_csv('submission_xgb.csv', index=False)

#%%

#LR part
from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import MinMaxScaler
# lr_scaler = MinMaxScaler()
# lr_scaler.fit(X_train)
# lr_train = lr_scaler.transform(X_train)
# lr_val = lr_scaler.transform(X_valid)
# lr_test = lr_scaler.transform(X_test)
#
# lr_model = LinearRegression(n_jobs=-1)
# lr_model.fit(lr_train, y_train)
#
# y_train_lr = lr_model.predict(X_train)
# y_valid_lr = lr_model.predict(X_valid)
# y_test_lr = lr_model.predict(X_test)
#
# from sklearn.metrics import mean_squared_error
# print('Train rmse:', np.sqrt(mean_squared_error(y_train, y_train_lr)))
# print('Validation rmse:', np.sqrt(mean_squared_error(y_valid, y_valid_lr)))
# np.savetxt('y_train_lr.csv',y_train_lr,delimiter=',',encoding='utf-8-sig',fmt='%.12f')
# np.savetxt('y_valid_lr.csv',y_valid_lr,delimiter=',',encoding='utf-8-sig',fmt='%.12f')
# np.savetxt('y_test_lr.csv',y_test_lr,delimiter=',',encoding='utf-8-sig',fmt='%.12f')
#
# submission = pd.read_csv('sample_submission.csv')
# submission['item_cnt_month'] = y_test_lr
# submission.to_csv('submission_lr.csv', index=False)

#%%

#RF part
# from sklearn.ensemble import RandomForestRegressor
# rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0, n_jobs=-1)
# rf_model.fit(X_train, y_train)
#
# y_train_rf = rf_model.predict(X_train)
# y_valid_rf = rf_model.predict(X_valid)
# y_test_rf = rf_model.predict(X_test)
#
# from sklearn.metrics import mean_squared_error
# print('Train rmse:', np.sqrt(mean_squared_error(y_train, y_train_rf)))
# print('Validation rmse:', np.sqrt(mean_squared_error(y_valid, y_valid_rf)))
# np.savetxt('y_train_rf.csv',y_train_rf,delimiter=',',encoding='utf-8-sig',fmt='%.12f')
# np.savetxt('y_valid_rf.csv',y_valid_rf,delimiter=',',encoding='utf-8-sig',fmt='%.12f')
# np.savetxt('y_test_rf.csv',y_test_rf,delimiter=',',encoding='utf-8-sig',fmt='%.12f')
#
# submission = pd.read_csv('sample_submission.csv')
# submission['item_cnt_month'] = y_test_rf
# submission.to_csv('submission_rf.csv', index=False)


#%%

combine_test = pd.DataFrame(y_test_lgb, columns=['lgb'])
# combine_test['lr']=y_test_lr
# combine_test['rf']=y_test_rf
combine_test['xgb']=y_test_xgb

combine_train = pd.DataFrame(y_train_lgb, columns=['lgb'])
# combine_train['lr']=y_train_lr
# combine_train['rf']=y_train_rf
combine_train['xgb']=y_train_xgb

combine_val = pd.DataFrame(y_valid_lgb, columns=['lgb'])
# combine_val['lr']=y_valid_lr
# combine_val['rf']=y_valid_rf
combine_val['xgb']=y_valid_xgb

combine_lr_model = LinearRegression(n_jobs=-1)
combine_lr_model.fit(combine_train, y_train)

y_train_com = combine_lr_model.predict(combine_train)
y_valid_com = combine_lr_model.predict(combine_val)
y_test_com = combine_lr_model.predict(combine_test)

from sklearn.metrics import mean_squared_error
print('Train rmse:', np.sqrt(mean_squared_error(y_train, y_train_com)))
print('Validation rmse:', np.sqrt(mean_squared_error(y_valid, y_valid_com)))

submission = pd.read_csv('sample_submission.csv')
submission['item_cnt_month'] = y_test_com
submission.to_csv('submission_com.csv', index=False)