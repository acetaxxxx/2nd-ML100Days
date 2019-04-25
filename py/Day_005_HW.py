
#%%
# Import 需要的套件
import os
import numpy as np
import pandas as pd

# 設定 data_path
dir_data = './data/'


#%%
f_app_train = os.path.join(dir_data, 'application_train.csv')
app_train = pd.read_csv(f_app_train)


#%%
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# ## 練習時間
#%% [markdown]
# 觀察有興趣的欄位的資料分佈，並嘗試找出有趣的訊息
# #### Eg
# - 計算任意欄位的平均數及標準差
# - 畫出任意欄位的[直方圖](https://zh.wikipedia.org/zh-tw/%E7%9B%B4%E6%96%B9%E5%9B%BE)
# 
# ### Hints:
# - [Descriptive Statistics For pandas Dataframe](https://chrisalbon.com/python/data_wrangling/pandas_dataframe_descriptive_stats/)
# - [pandas 中的繪圖函數](https://amaozhao.gitbooks.io/pandas-notebook/content/pandas%E4%B8%AD%E7%9A%84%E7%BB%98%E5%9B%BE%E5%87%BD%E6%95%B0.html)
# 

#%%


data_income = app_train['AMT_INCOME_TOTAL']
mean_income = app_train['AMT_INCOME_TOTAL'].mean()
std_income = app_train['AMT_INCOME_TOTAL'].std()

normalize_income = (data_income-mean_income)/std_income
color = dict(boxes='Green', whiskers='Orange',
                medians='Blue', caps='White')
data_income.plot(kind='box',color=color,vert=False)

#%% markdown
# ## 從上面的圖看出這個資料有很明顯的極端值
# ## 先不移除換另外一個變數

#%%

data_credit = app_train['AMT_CREDIT']
mean_Credit = data_credit.mean()
std_Credit = data_credit.std()

print(mean_Credit)
print(std_Credit)
data_credit.plot(kind='box',color=color,vert=False)

#data_credit.plot(kind='hist')

#%%

plt.hist(x=data_credit,histtype='step')

#%%
