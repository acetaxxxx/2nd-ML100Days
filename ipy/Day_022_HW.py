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
# - 試著模仿範例寫法, 在鐵達尼生存預測中, 觀察標籤編碼與獨編碼熱的影響
#%% [markdown]
# # [作業重點]
# - 回答在範例中的觀察結果
# - 觀察標籤編碼與獨熱編碼, 在特徵數量 / 邏輯斯迴歸分數 / 邏輯斯迴歸時間上, 分別有什麼影響 (In[3], Out[3], In[4], Out[4]) 
#%% [markdown]
# # 作業1
# * 觀察範例，在房價預測中調整標籤編碼(Label Encoder) / 獨熱編碼 (One Hot Encoder) 方式，  
# 對於線性迴歸以及梯度提升樹兩種模型，何者影響比較大?
# 線性回歸影響比較大 看得出來分處的差距頗大的

#%%
# 做完特徵工程前的所有準備 (與前範例相同)
import pandas as pd
import numpy as np
import copy, time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

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
print(f'{len(object_features)} Numeric Features : {object_features}\n')

# 只留類別型欄位
df = df[object_features]
df = df.fillna('None')
train_num = train_Y.shape[0]
df.head()

#%% [markdown]
# # 作業2
# * 鐵達尼號例題中，標籤編碼 / 獨熱編碼又分別對預測結果有何影響? (Hint : 參考今日範例)
#  羅吉斯回歸是用來分類的 因此使用 OneHotEncoder 會比較明顯
#%%
# 標籤編碼 + 羅吉斯迴歸
df_temp = pd.DataFrame()
for e in object_features:
    df_temp[e] = LabelEncoder().fit_transform(df[e])

train_X = df_temp[:train_num]
estimator = LogisticRegression()
start = time.time()
print(f'shape : {train_X.shape}')
print(f'score : {cross_val_score(estimator, train_X, train_Y, cv=5).mean()}')
print(f'time : {time.time() - start} sec')



#%%
# 獨熱編碼 + 羅吉斯迴歸
df_temp = pd.get_dummies(df)
train_X = df_temp[:train_num]
estimator = LogisticRegression()
start = time.time()
print(f'shape : {train_X.shape}')
print(f'score : {cross_val_score(estimator, train_X, train_Y, cv=5).mean()}')
print(f'time : {time.time() - start} sec')



#%%



