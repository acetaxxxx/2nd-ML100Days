#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'ipy'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # [作業目標]
# - 使用 Day 12 剛學到的方法, 對較完整的資料生成離散化特徵
# - 觀察上述離散化特徵, 對於目標值的預測有沒有幫助
#%% [markdown]
# # [作業重點]
# - 仿照 Day 12 的語法, 將年齡資料 ('DAYS_BIRTH' 除以 365) 離散化
# - 繪製上述的 "離散化標籤" 與目標值 ('TARGET') 的長條圖

#%%
# 載入需要的套件
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# 設定 data_path
dir_data = './data/'

#%% [markdown]
# ### 之前做過的處理

#%%
# 讀取資料檔
f_app_train = os.path.join(dir_data, 'application_train.csv')
app_train = pd.read_csv(f_app_train)
app_train.shape


#%%
# 將只有兩種值的類別型欄位, 做 Label Encoder, 計算相關係數時讓這些欄位可以被包含在內
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# 檢查每一個 column
for col in app_train:
    if app_train[col].dtype == 'object':
        # 如果只有兩種值的類別型欄位
        if len(list(app_train[col].unique())) <= 2:
            # 就做 Label Encoder, 以加入相關係數檢查
            app_train[col] = le.fit_transform(app_train[col])            
print(app_train.shape)
app_train.head()


#%%
# 受雇日數為異常值的資料, 另外設一個欄位記錄, 並將異常的日數轉成空值 (np.nan)
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

# 出生日數 (DAYS_BIRTH) 取絕對值 
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])


#%% [markdown]
# ## 練習時間
# 參考 Day 12 範例程式，離散化你覺得有興趣的欄位，並嘗試找出有趣的訊息

#%%

app_train['BASEMENTAREA_MODE'].replace({np.nan:0.0},inplace = True)

app_train['BASEMENTAREA_MODE'].head(50)

app_train['ABS_DAYS_REGISTRATION'] = app_train['DAYS_REGISTRATION'].apply(lambda x: abs(x))

app_train['ABS_DAYS_REGISTRATION'].head(3)



#%%
app_train['DIS_ABS_DAY_REGISTRATION'] = pd.cut(app_train['ABS_DAYS_REGISTRATION'],10)

basement_grouped = app_train.groupby('DIS_ABS_DAY_REGISTRATION')['BASEMENTAREA_MODE']

basement_grouped.describe()

#%%

app_train['BASEMENT_Z'] = basement_grouped.apply(lambda x : (x-x.mean())/x.std())

app_train['BASEMENT_Z'].head()

#%%

plt.scatter(x=app_train['BASEMENT_Z'],y=app_train['ABS_DAYS_REGISTRATION'])
plt.show()



#%%
