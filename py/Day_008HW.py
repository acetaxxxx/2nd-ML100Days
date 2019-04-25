#%% [markdown]
# # 常用的 DataFrame 操作
# * merge / transform
# * subset
# * groupby

#%%
# Import 需要的套件
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


#%%
# 設定 data_path
dir_data = './data/'


#%%
f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)
app_train.head()

#%% [markdown]
# ## 作業
# 1. 請將 app_train 中的 CNT_CHILDREN 依照下列規則分為四組，並將其結果在原本的 dataframe 命名為 CNT_CHILDREN_GROUP
#     * 0 個小孩
#     * 有 1 - 2 個小孩
#     * 有 3 - 5 個小孩
#     * 有超過 5 個小孩
# 
# 2. 請根據 CNT_CHILDREN_GROUP 以及 TARGET，列出各組的平均 AMT_INCOME_TOTAL，並繪製 boxplot
# 3. 請根據 CNT_CHILDREN_GROUP 以及 TARGET，對 AMT_INCOME_TOTAL 計算 [Z 轉換](https://en.wikipedia.org/wiki/Standard_score) 後的分數

#%%
#1
"""
Your code here
"""
cut_rule = [0,1,3,6,float("inf")]

app_train['CNT_CHILDREN_GROUP'] = pd.cut(app_train['CNT_CHILDREN'].values, cut_rule, include_lowest=True,right=False)
app_train['CNT_CHILDREN_GROUP'].value_counts()


#%%
#2-1
"""
Your code here
"""
grp = 'CNT_CHILDREN_GROUP'

grouped_df = app_train.groupby(grp)['AMT_INCOME_TOTAL']

grouped_df.mean()




#%%
#2-2
"""
Your code here
"""
plt_column = 'AMT_INCOME_TOTAL'
plt_by = 'CNT_CHILDREN_GROUP'

app_train.boxplot(column=plt_column, by = plt_by, showfliers = False, figsize=(12,12))
plt.suptitle('')
plt.show()


#%%
#3
"""
Your code here
"""
app_train['AMT_INCOME_TOTAL_Z_BY_CHILDREN_GRP-TARGET'] = grouped_df.apply(lambda x: (x-x.mean())/x.std())

app_train[['AMT_INCOME_TOTAL','AMT_INCOME_TOTAL_Z_BY_CHILDREN_GRP-TARGET']].head()




#%%
