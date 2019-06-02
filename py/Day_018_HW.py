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
#%% [markdown]
# # [作業目標]
# - 試著完成三種不同特徵類型的三種資料操作, 觀察結果
# - 思考一下, 這三種特徵類型, 哪一種應該最複雜/最難處理
#%% [markdown]
# # [作業重點]
# - 完成剩餘的八種 類型 x 操作組合 (In[6]~In[13], Out[6]~Out[13])
# - 思考何種特徵類型, 應該最複雜

#%%
# 載入基本套件
import pandas as pd
import numpy as np

# 讀取訓練與測試資料
data_path = 'data/'
df_train = pd.read_csv(data_path + 'titanic_train.csv')
df_test = pd.read_csv(data_path + 'titanic_test.csv')
df_train.shape


#%%
# 重組資料成為訓練 / 預測用格式
train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])
df.head()


#%%
# 秀出資料欄位的類型與數量
dtype_df = df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df = dtype_df.groupby("Column Type").aggregate('count').reset_index()
dtype_df


#%%
#確定只有 int64, float64, object 三種類型後, 分別將欄位名稱存於三個 list 中
int_features = []
float_features = []
object_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64':
        float_features.append(feature)
    elif dtype == 'int64':
        int_features.append(feature)
    else:
        object_features.append(feature)
print(f'{len(int_features)} Integer Features : {int_features}\n')
print(f'{len(float_features)} Float Features : {float_features}\n')
print(f'{len(object_features)} Object Features : {object_features}')

#%% [markdown]
# # 作業1 
# * 試著執行作業程式，觀察三種類型 (int / float / object) 的欄位分別進行( 平均 mean / 最大值 Max / 相異值 nunique )  
# 中的九次操作會有那些問題? 並試著解釋那些發生Error的程式區塊的原因?  
# 
# > object 會有問題 因為他的mean() 無法計算 Max()也是同樣的道理 不過它已經轉型成為string去做比較
# # 作業2
# * 思考一下，試著舉出今天五種類型以外的一種或多種資料類型，你舉出的新類型是否可以歸在三大類中的某些大類?  
# 所以三大類特徵中，哪一大類處理起來應該最複雜?
# > 類別型應該是會最為麻煩 因為無法肯定資料轉換後的數字代表的意義
#%%
# 例 : 整數 (int) 特徵取平均 (mean)
df[int_features].mean()


#%%
# 請依序列出 三種特徵類型 (int / float / object) x 三種方法 (平均 mean / 最大值 Max / 相異值 nunique) 的其餘操作
"""
Your Code Here
"""
df[int_features].max()




#%%
df[int_features].nunique()


#%%
df[float_features].mean()


#%%
df[float_features].max()

#%%
df[float_features].nunique()

#%%
df[object_features].mean()


#%%
df[object_features].max()

#%%
df[object_features].nunique()

#%%
