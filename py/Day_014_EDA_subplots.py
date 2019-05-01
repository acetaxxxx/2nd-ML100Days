# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import warnings
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'ipy'))
    print(os.getcwd())
except:
    pass
# %% [markdown]
# # [Plot] Subplots
# * 將塞太多組別的圖，拆成多張檢視
# %% [markdown]
# # [教學目標]
# - 以下程式碼將示範如何將多張圖形, 使用 Subplot 與其參數排定顯示相對位置
# %% [markdown]
# # [範例重點]
# - 傳統的 subplot 三碼 (row,column,idx) 繪製法 (In[6], Out[6])
# - subplot index 超過 10 以上的繪圖法 (In[7], Out[7])

# %%
# 載入需要的套件
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # 另一個繪圖-樣式套件

# 忽略警告訊息
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('dark_background')
warnings.filterwarnings('ignore')

# 設定 data_path
dir_data = './data/'


# %%
# 讀取檔案
f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)
app_train.head()


# %%
# 資料整理 ( 'DAYS_BIRTH'全部取絕對值 )
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])


# %%
# 根據年齡分成不同組別 (年齡區間 - 還款與否)
age_data = app_train[['TARGET', 'DAYS_BIRTH']]  # subset
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365  # day-age to year-age

# 連續資料離散化
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'],
                                  bins=np.linspace(20, 70, num=11))  # 自 20 到 70 歲，切 11 個點 (得到 10 組)
print(age_data['YEARS_BINNED'].value_counts())
age_data.head()


# %%
# 資料分群後排序
year_group_sorted = np.sort(age_data['YEARS_BINNED'].unique())
age_data.head()

# 繪製分群後的 10 條 KDE 曲線
plt.figure(figsize=(8, 6))
for i in range(len(year_group_sorted)):
    sns.distplot(age_data.loc[(age_data['YEARS_BINNED'] == year_group_sorted[i]) & (
        age_data['TARGET'] == 0), 'YEARS_BIRTH'], label=str(year_group_sorted[i]))

    sns.distplot(age_data.loc[(age_data['YEARS_BINNED'] == year_group_sorted[i]) & (
        age_data['TARGET'] == 1), 'YEARS_BIRTH'], label=str(year_group_sorted[i]))
plt.title('KDE with Age groups')
plt.show()

# %% [markdown]
# ## Subplot
# plt.subplot(row,column,idx)

# %%
# 每張圖大小為 8x8
plt.figure(figsize=(8, 8))

# plt.subplot 三碼如上所述, 分別表示 row總數, column總數, 本圖示第幾幅(idx)
plt.subplot(321)
plt.plot([0, 1], [0, 1], label='I am subplot1')
plt.legend()

plt.subplot(322)
plt.plot([0, 1], [1, 0], label='I am subplot2')
plt.legend()

plt.subplot(323)
plt.plot([1, 0], [0, 1], label='I am subplot3')
plt.legend()

plt.subplot(324)
plt.plot([1, 0], [1, 0], label='I am subplot4')
plt.legend()

plt.subplot(325)
plt.plot([0, 1], [0.5, 0.5], label='I am subplot5')
plt.legend()

plt.subplot(326)
plt.plot([0.5, 0.5], [0, 1], label='I am subplot6')
plt.legend()

plt.show()


# %%
# subplot index 超過10以上的繪製方式
nrows = 5
ncols = 2

plt.figure(figsize=(10, 30))
fig, axis = plt.subplots(ncols=ncols, nrows=nrows)
fig.set_size_inches(10,30)
for i in range(len(year_group_sorted)):
    plt.subplot(nrows, ncols, i+1)
    axis[int(i/ncols), int(i % ncols)] = sns.distplot(age_data.loc[(age_data['YEARS_BINNED'] == year_group_sorted[i]) & (age_data['TARGET'] == 0), 'YEARS_BIRTH'],
                                                      label="TARGET = 0", hist=False)
    axis[int(i/ncols), int(i % ncols)] = sns.distplot(age_data.loc[(age_data['YEARS_BINNED'] == year_group_sorted[i]) & (age_data['TARGET'] == 1), 'YEARS_BIRTH'],
                                                      label="TARGET = 1", hist=False)
    plt.title(str(year_group_sorted[i]))
plt.show()

# %% [markdown]
# ## 作業
# ### 請使用 application_train.csv, 根據不同的 HOUSETYPE_MODE 對 AMT_CREDIT 繪製 Histogram
