#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'ipy'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # 作業 : (Kaggle)房價預測
#%% [markdown]
# # [作業目標]
# - 試著模仿範例寫法, 在房價預測中, 觀察去除離群值的影響
#%% [markdown]
# # [作業重點]
# - 觀察將極端值以上下限值取代, 對於分布與迴歸分數的影響 (In[5], Out[5])
# - 觀察將極端值資料直接刪除, 對於分布與迴歸分數的影響 (In[6], Out[6])

#%%
# 做完特徵工程前的所有準備 (與前範例相同)
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

data_path = 'data/'
df_train = pd.read_csv(data_path + 'house_train.csv.gz')

train_Y = np.log1p(df_train['SalePrice'])
df = df_train.drop(['Id', 'SalePrice'] , axis=1)
df.head()


#%%
#只取 int64, float64 兩種數值型欄位, 存於 num_features 中
num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)
print(f'{len(num_features)} Numeric Features : {num_features}\n')


#%%
# 削減文字型欄位, 只剩數值型欄位
df = df[num_features]
df = df.fillna(-1)
MMEncoder = MinMaxScaler()
train_num = train_Y.shape[0]
df.head()

#%% [markdown]
# # 作業1
# * 試著限制 '1樓地板面積(平方英尺)' (1stFlrSF) 欄位的上下限, 看看能否再進一步提高分數?

#%%
# 顯示 1stFlrSF 與目標值的散佈圖
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('dark_background')
sns.regplot(x = df['1stFlrSF'][:train_num], y=train_Y)
plt.show()

# 做線性迴歸, 觀察分數
train_X = MMEncoder.fit_transform(df)
estimator = LinearRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()


#%%
# 將 1stFlrSF 限制在你覺得適合的範圍內, 調整離群值

df['1stFlrSF'] = df['1stFlrSF'].clip(500,2500)

sns.regplot(x = df['1stFlrSF'][:train_num], y=train_Y)
plt.show()
# 做線性迴歸, 觀察分數
train_X = MMEncoder.fit_transform(df)
estimator = LinearRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()

#%% [markdown]
# # 作業2
# * 續前題, 去除離群值有兩類方式 :  捨棄離群值(刪除離群的資料) 以及調整離群值,  
# 請試著用同樣的上下限, 改為 '捨棄離群值' 的方法, 看看結果會變好還是變差? 並試著解釋原因。

#%%
# 將 1stFlrSF 限制在你覺得適合的範圍內, 捨棄離群值
df_train = pd.read_csv(data_path + 'house_train.csv.gz')

train_Y = np.log1p(df_train['SalePrice'])
df = df_train.drop(['Id', 'SalePrice'] , axis=1)
df = df[num_features]
df = df.fillna(-1)
MMEncoder = MinMaxScaler()
train_num = train_Y.shape[0]


df = df[ (df['1stFlrSF']>500) & (df['1stFlrSF']<2500)]
df
#%%
df.index
#%%
# 做線性迴歸, 觀察分數
train_Y  = train_Y[df.index]
sns.regplot(x = df['1stFlrSF'][:train_num], y=train_Y)
plt.show()
train_X = MMEncoder.fit_transform(df)
estimator = LinearRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()




#%%
# 結果會變好 因為其實離群值會還是算一個資料點