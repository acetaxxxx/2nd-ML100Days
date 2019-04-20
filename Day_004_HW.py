
#%%
import os
import numpy as np
import pandas as pd


#%%
# 設定 data_path, 並讀取 app_train
dir_data = './data/'
f_app_train = os.path.join(dir_data, 'application_train.csv')
app_train = pd.read_csv(f_app_train)

#%% [markdown]
# ## 作業
# 將下列部分資料片段 sub_train 使用 One Hot encoding, 並觀察轉換前後的欄位數量 (使用 shape) 與欄位名稱 (使用 head) 變化

#%%
sub_train = pd.DataFrame(app_train['WEEKDAY_APPR_PROCESS_START'])
print(sub_train.shape)
sub_train.head()


#%%
"""
Your Code Here
"""
after_dummies = pd.get_dummies(sub_train)
print(after_dummies.shape)
after_dummies.head()



#%%
