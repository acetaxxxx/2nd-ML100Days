#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'ipy'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # 作業 : (Kaggle)鐵達尼生存預測
# https://www.kaggle.com/c/titanic
#%% [markdown]
# # [作業目標]
# - 試著模仿範例寫法, 在鐵達尼生存預測中, 觀察計數編碼與特徵雜湊的效果
#%% [markdown]
# # [作業重點]
# - 仿造範例, 完成計數編碼以及搭配邏輯斯迴歸的預測 (In[4], Out[4], In[5], Out[5]) 
# - 仿造範例, 完成雜湊編碼, 以及計數編碼+雜湊編碼 搭配邏輯斯迴歸的預測 (In[6], Out[6], In[7], Out[7]) 
# - 試著回答上述執行結果的觀察
#%% [markdown]
# # 作業1
# * 參考範例，將鐵達尼的艙位代碼( 'Cabin' )欄位使用特徵雜湊 / 標籤編碼 / 目標均值編碼三種轉換後，  
# 與其他類別型欄位一起預估生存機率

#%%
# 做完特徵工程前的所有準備 (與前範例相同)
import pandas as pd
import numpy as np
import copy, time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

data_path = 'data/'
df_train = pd.read_csv(data_path + 'titanic_train.csv')
df_test = pd.read_csv(data_path + 'titanic_test.csv')

train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])
df.head()


#%%
#只取類別值 (object) 型欄位, 存於 object_features 中
object_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'object':
        object_features.append(feature)
print(f'{len(object_features)} Object Features : {object_features}\n')

# 只留類別型欄位
df = df[object_features]
df = df.fillna('None')
train_num = train_Y.shape[0]
df.head()

#%% [markdown]
# # 作業2
# * 承上題，三者比較效果何者最好?

#%%
# 對照組 : 標籤編碼 + 邏輯斯迴歸
df_temp = pd.DataFrame()
for c in df.columns:
    df_temp[c] = LabelEncoder().fit_transform(df[c])
train_X = df_temp[:train_num]
estimator = LogisticRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())
df_temp.head()


#%%
# 加上 'Cabin' 欄位的計數編碼
df_group  = df_temp.groupby(['Cabin'])['Name'].agg({'Cabin_Cnt':'size'}).reset_index()
#df_group
df_temp = df_temp.merge(df_group,how='left',on=['Cabin'])
df_temp




#%%
# 'Cabin'計數編碼 + 邏輯斯迴歸
train_X = df_temp[:train_num]
estimator = LogisticRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())
df_temp.head()






#%%
# 'Cabin'特徵雜湊 + 邏輯斯迴歸
df_temp = pd.DataFrame()
for c in df.columns:
    df_temp[c] = LabelEncoder().fit_transform(df[c])
df_temp['Cabin_Hash'] = df.reset_index()['Cabin'].map(lambda x : hash(x) % 3)
train_X = df_temp[:train_num]
estimator = LogisticRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())
df_temp.head()
#%%






#%%
# 'Cabin'計數編碼 + 'Cabin'特徵雜湊 + 邏輯斯迴歸
df_group  = df_temp.groupby(['Cabin'])['Name'].agg({'Cabin_Cnt':'size'}).reset_index()
#df_group
df_temp = df_temp.merge(df_group,how='left',on=['Cabin'])
train_X = df_temp[:train_num]
estimator = LogisticRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())
df_temp.head()
#%%



