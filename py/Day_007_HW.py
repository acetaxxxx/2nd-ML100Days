#%% [markdown]
# # 處理 outliers
# * 新增欄位註記
# * outliers 或 NA 填補
#     1. 平均數 (mean)
#     2. 中位數 (median, or Q50)
#     3. 最大/最小值 (max/min, Q100, Q0)
#     4. 分位數 (quantile)

#%%
# Import 需要的套件
import os
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

# 設定 data_path
dir_data = './data/'


#%%
f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)
app_train.head()

#%% [markdown]
# ## 1. 列出 AMT_ANNUITY 的 q0 - q100
# ## 2.1 將 AMT_ANNUITY 中的 NAs 暫時以中位數填補
# ## 2.2 將 AMT_ANNUITY 的數值標準化至 -1 ~ 1 間
# ## 3. 將 AMT_GOOD_PRICE 的 NAs 以眾數填補
# 

#%%
"""
YOUR CODE HERE
"""
# 1: 計算 AMT_ANNUITY 的 q0 - q100
q_all = [app_train['AMT_ANNUITY'].quantile(q=i/100) for i in range(101)]

pd.DataFrame({'q': list(range(101)),
              'value': q_all})


#%%
# 2.1 將 NAs 以 q50 填補
print("Before replace NAs, numbers of row that AMT_ANNUITY is NAs: %i" % sum(app_train['AMT_ANNUITY'].isnull()))

"""
Your Code Here
"""
q_50 = app_train['AMT_ANNUITY'].quantile(q=50/100)
app_train.loc[app_train['AMT_ANNUITY'].isnull(),'AMT_ANNUITY'] = q_50

print("After replace NAs, numbers of row that AMT_ANNUITY is NAs: %i" % sum(app_train['AMT_ANNUITY'].isnull()))

#%% [markdown]
# ### Hints: Normalize function (to -1 ~ 1)
# $ y = 2*(\frac{x - min(x)}{max(x) - min(x)} - 0.5) $

#%%
# 2.2 Normalize values to -1 to 1
print("== Original data range ==")
print(app_train['AMT_ANNUITY'].describe())

def normalize_value(x):
    """
    Your Code Here, compelete this function
    """
    _mean  = x.mean()
    _std = x.std()
    x = (x-_mean)/_std
    return x

app_train['AMT_ANNUITY_NORMALIZED'] = normalize_value(app_train['AMT_ANNUITY'])

print("== Normalized data range ==")
app_train['AMT_ANNUITY_NORMALIZED'].describe()


#%%
# 3
print("Before replace NAs, numbers of row that AMT_GOODS_PRICE is NAs: %i" % sum(app_train['AMT_GOODS_PRICE'].isnull()))

# 列出重複最多的數值
"""
Your Code Here
"""

value_most = scipy.stats.mode(app_train['AMT_GOODS_PRICE'])
print(value_most)

mode_goods_price = list(app_train['AMT_GOODS_PRICE'].value_counts().index)
app_train.loc[app_train['AMT_GOODS_PRICE'].isnull(), 'AMT_GOODS_PRICE'] = mode_goods_price[0]

print("After replace NAs, numbers of row that AMT_GOODS_PRICE is NAs: %i" % sum(app_train['AMT_GOODS_PRICE'].isnull()))




#%%
