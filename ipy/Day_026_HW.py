#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'ipy'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # 範例 : 計程車費率預測
# https://www.kaggle.com/c/new-york-city-taxi-fare-prediction
#%% [markdown]
# # [作業目標]
# - 使用並觀察特徵組合, 在計程車費率預測競賽的影響
#%% [markdown]
# # [作業重點]
# - 仿造範例並參考今日課程內容, 使用經緯度一圈的長度比的概念造出新特徵, 觀察有什麼影響 (In[6], Out[6])
# - 只使用上面所造的這個新特徵, 觀察有什麼影響 (In[7], Out[7])

#%%
# 做完特徵工程前的所有準備
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

data_path = 'data/'
df = pd.read_csv(data_path + 'taxi_data1.csv')

train_Y = df['fare_amount']
df = df.drop(['fare_amount'] , axis=1)
df.head()


#%%
# 時間特徵分解方式:使用datetime
df['pickup_datetime'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S UTC'))
df['pickup_year'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%Y')).astype('int64')
df['pickup_month'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%m')).astype('int64')
df['pickup_day'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%d')).astype('int64')
df['pickup_hour'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%H')).astype('int64')
df['pickup_minute'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%M')).astype('int64')
df['pickup_second'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%S')).astype('int64')
df.head()


#%%
# 將結果使用線性迴歸 / 梯度提升樹分別看結果
df = df.drop(['pickup_datetime'] , axis=1)
scaler = MinMaxScaler()
train_X = scaler.fit_transform(df)
Linear = LinearRegression()
print(f'Linear Reg Score : {cross_val_score(Linear, train_X, train_Y, cv=5).mean()}')
GDBT = GradientBoostingRegressor()
print(f'Gradient Boosting Reg Score : {cross_val_score(GDBT, train_X, train_Y, cv=5).mean()}')


#%%

# 增加緯度差, 經度差, 座標距離等三個特徵
df['longitude_diff'] = df['dropoff_longitude'] - df['pickup_longitude']
df['latitude_diff'] = df['dropoff_latitude'] - df['pickup_latitude']
df['distance_2D'] = (df['longitude_diff']**2 + df['latitude_diff']**2)**0.5
df[['distance_2D', 'longitude_diff', 'latitude_diff', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].head()


#%%
# 結果 : 準確度上升
train_X = scaler.fit_transform(df)
print(f'Linear Reg Score : {cross_val_score(Linear, train_X, train_Y, cv=5).mean()}')
print(f'Gradient Boosting Reg Score : {cross_val_score(GDBT, train_X, train_Y, cv=5).mean()}')

#%% [markdown]
# # 作業1
# * 參考今日教材，試著使用經緯度一圈的長度比這一概念，組合出一個新特徵，再觀察原特徵加上新特徵是否提升了正確率?

#%%
from math import sin, cos, sqrt, atan2, radians
R = 6373.0
# df.apply(lambda x : print(x['pickup_latitude']),axis = 1)
df['distance_temp']=df.apply(lambda x: sin(radians(x['latitude_diff']) / 2)**2 + cos(radians(x['pickup_latitude'])) * cos(radians(x['dropoff_latitude'])) * sin(radians(x['longitude_diff']) / 2)**2,axis = 1)

df['distance_real']=df.apply(lambda x : R * 2 * atan2(sqrt(x['distance_temp']), sqrt(1 - x['distance_temp'])),axis = 1)
"""
Your Code Here, set new character at df['distance_real']
"""
df.drop(columns=['distance_temp'],inplace=True)
# 觀察結果 
train_X = scaler.fit_transform(df)
print(f'Linear Reg Score : {cross_val_score(Linear, train_X, train_Y, cv=5).mean()}')
print(f'Gradient Boosting Reg Score : {cross_val_score(GDBT, train_X, train_Y, cv=5).mean()}')

#%% [markdown]
# # 作業2
# * 試著只使用新特徵估計目標值(忽略原特徵)，效果跟作業1的結果比較起來效果如何?
df.drop(columns=['distance_2D','latitude_diff','longitude_diff','pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude'])
#%%
train_X = scaler.fit_transform(df[['distance_real']])
print(f'Linear Reg Score : {cross_val_score(Linear, train_X, train_Y, cv=5).mean()}')
print(f'Gradient Boosting Reg Score : {cross_val_score(GDBT, train_X, train_Y, cv=5).mean()}')


#%%



