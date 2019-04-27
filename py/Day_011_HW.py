#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'ipy'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # [EDA] 了解變數分布狀態: Bar & KDE (density plot)
#%% [markdown]
# # To do: 變項的分群比較
# 1. 自 20 到 70 歲，切 11 個點，進行分群比較 (KDE plot)
# 2. 以年齡區間為 x, target 為 y 繪製 barplot
#%% [markdown]
# # [作業目標]
# - 試著調整資料, 並利用提供的程式繪製分布圖
#%% [markdown]
# # [作業重點]
# - 如何將資料依照歲數, 將 20 到 70 歲切成11個區間? (In[4], Hint : 使用 numpy.linspace),  
#   送入繪圖前的除了排序外, 還要注意什麼? (In[5])
# - 如何調整對應資料, 以繪製長條圖(bar chart)? (In[7])

#%%
# 載入需要的套件
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # 另一個繪圖-樣式套件
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('dark_background')

# 忽略警告訊息
import warnings
warnings.filterwarnings('ignore')

# 設定 data_path
dir_data = './data/'


#%%
# 讀取檔案
print(os.getcwd())
f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)
app_train.head()


#%%
# 資料整理 ( 'DAYS_BIRTH'全部取絕對值 )
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])


#%%
# 根據年齡分成不同組別 (年齡區間 - 還款與否)
age_data = app_train[['TARGET', 'DAYS_BIRTH']] # subset
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365 # day-age to year-age

#自 20 到 70 歲，切 11 個點 (得到 10 組)
# """
# Your Code Here
#  """
bin_cut =  [ 20+(50/11)*i for i in range(12)]
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = bin_cut) 

# 顯示不同組的數量
print(age_data['YEARS_BINNED'].value_counts())
age_data.head()


#%%
year_group_sorted = age_data.sort_values('YEARS_BINNED')['YEARS_BINNED'].unique()
print(year_group_sorted)
#%%
# 繪圖前先排序 / 分組
"""
Your Code Here
"""

year_group_sorted = age_data.sort_values('YEARS_BINNED')['YEARS_BINNED'].unique()

plt.figure(figsize=(8,6))
for i in range(len(year_group_sorted)):
    sns.distplot(age_data.loc[(age_data['YEARS_BINNED'] == year_group_sorted[i]) &(age_data['TARGET'] == 0), 'YEARS_BIRTH'], label = str(year_group_sorted[i]))
    
    sns.distplot(age_data.loc[(age_data['YEARS_BINNED'] == year_group_sorted[i]) & (age_data['TARGET'] == 1), 'YEARS_BIRTH'], label = str(year_group_sorted[i]))
plt.title('KDE with Age groups')
plt.show()


#%%
# 計算每個年齡區間的 Target、DAYS_BIRTH與 YEARS_BIRTH 的平均值
age_groups  = age_data.groupby('YEARS_BINNED').mean()
age_groups


#%%
plt.figure(figsize = (8, 8))

# 以年齡區間為 x, target 為 y 繪製 barplot
"""
Your Code Here
"""
px = list(age_data['YEARS_BINNED'].unique())
py = age_groups['TARGET']
sns.barplot(px, py)

# Plot labeling
plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group');




#%%
