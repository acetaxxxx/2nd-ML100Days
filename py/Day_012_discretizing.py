#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'ipy'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # [教學目標]
# - 以下程式碼將示範在 python 如何利用 pandas.cut 與 .qcut 計算出數據的離散化標籤
#%% [markdown]
# # [範例重點]
# - pandas.cut 的等寬劃分效果 (In[3], Out[4])
# - pandas.qcut 的等頻劃分效果 (In[5], Out[6])

#%%
# 載入套件
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
# 初始設定 Ages 的資料
ages = pd.DataFrame({"age": [18,22,25,27,7,21,23,37,30,61,45,41,9,18,80,100]})

#%% [markdown]
# #### 等寬劃分

#%%
# 新增欄位 "equal_width_age", 對年齡做等寬劃分
ages["equal_width_age"] = pd.cut(ages["age"], 4)


#%%
# 觀察等寬劃分下, 每個種組距各出現幾次
ages["equal_width_age"].value_counts() # 每個 bin 的值的範圍大小都是一樣的

#%% [markdown]
# #### 等頻劃分

#%%
# 新增欄位 "equal_freq_age", 對年齡做等頻劃分
ages["equal_freq_age"] = pd.qcut(ages["age"], 4)


#%%
# 觀察等頻劃分下, 每個種組距各出現幾次
ages["equal_freq_age"].value_counts() # 每個 bin 的資料筆數是一樣的

#%% [markdown]
# ### 作業
# 新增一個欄位 `customized_age_grp`，把 `age` 分為 (0, 10], (10, 20], (20, 30], (30, 50], (50, 100] 這五組，'(' 表示不包含, ']' 表示包含
# 
# Hints: 執行 ??pd.cut()，了解提供其中 bins 這個參數的使用方式

#%%



