#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('HDFCBANK.csv')
df.head()


# In[3]:


fig, ax = plt.subplots(figsize=(10,10))  
dataplot = sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)


# In[4]:


data = df[['Date','Open','High','Low','Volume','VWAP']]


# In[5]:


df.set_index("Date", drop=False, inplace=True)


# In[6]:


df.Close.plot(figsize=(14, 7))


# In[7]:


data.info()


# In[8]:


df.reset_index(drop=True, inplace=True)
lag_features = ["Open","High", "Low","VWAP"]
window1 = 3
window2 = 7

df_rolled_3d = df[lag_features].rolling(window=window1, min_periods=0)
df_mean_3d = df_rolled_3d.mean().shift(1).reset_index().astype(np.float32)

df_rolled_7d = df[lag_features].rolling(window=window2, min_periods=0)
df_std_7d = df_rolled_7d.std().shift(1).reset_index().astype(np.float32)

for feature in lag_features:
    df[f"{feature}_mean_lag{window1}"] = df_mean_3d[feature].astype('float32')
    df[f"{feature}_std_lag{window2}"] = df_std_7d[feature].astype('float32')


# In[9]:


df.fillna(df.mean(), inplace=True)


# In[10]:


df.set_index("Date", drop=False, inplace=True)
df.head()


# In[11]:


df.Date = pd.to_datetime(df.Date, format="%Y-%m-%d")
df["month"] = df.Date.dt.month
df["day"] = df.Date.dt.day
df["day_of_week"] = df.Date.dt.dayofweek
df.head()


# In[ ]:





# In[12]:


#EDA


# In[13]:


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['figure.figsize'] =(9,5)
matplotlib.rcParams['figure.facecolor']= '#00000000'


# In[14]:


start_date = "2019-04-30"
end_date = "2021-04-30"

mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
df1 =df.loc[mask]

df1


# In[15]:


df1['Open'].plot()
plt.title("HDFC BANK Open Price April-2019-2021")
plt.show


# In[16]:


df1['Close'].plot()
plt.title("HDFC BANK Close Price April-2019 - 2021")
plt.show


# In[17]:


AvgHigh = df1.describe()['High']['mean']
AvgLow = df1.describe()['Low']['mean']
AvgVolume = df1.describe()['Volume']['mean']
AvgDeliverable_volume=df1.describe()['Deliverable Volume']['mean']

print("Average High HDFC BANK Stock: ",AvgHigh,
     "\nAverage Low HDFC BANK: ",AvgLow,
     "\nAverage Volume for HDFC BANK: ", AvgVolume,
     "\nAverage Deliverable Volume of HDFC BANK: ", AvgDeliverable_volume)


# In[18]:


plt.figure(figsize=(20,7));
df1['Open'].plot(label = 'HDFC BANK Open Price')
df1['Close'].plot(label = 'HDFC BANK Close Price')
df1['High'].plot(label= 'HDFC BANK High Price')
df1['Low'].plot(label= 'HDFC BANK Low Price')
plt.legend()
plt.title('HDFC BANK Prices')
plt.ylabel("Stock Price")
plt.show()


# In[19]:


plt.figure(figsize=(20,12));
plt.plot(df1.Date, df1.High);
plt.plot(df1.Date, df1.Low);
plt.title("Trends of HDFC BANK Stock 2020-2021");
plt.xlabel('2019-2021');
plt.ylabel('Price in INR');
plt.legend(['High','Low']);


# In[20]:


plt.hist(df1.VWAP,bins=np.arange(750,2600,10));


# In[21]:


df1['Volume'].plot(figsize=(17,5))
plt.title("HDFC BANK Volume 2019-2021")
plt.show()


# In[22]:


df1.iloc[[df1['Volume'].argmax()]]


# In[23]:


spike = df1.iloc[270:285]
spike['Open'].plot()


# In[24]:


df1.iloc[265:300]['Open'].plot()


# In[25]:


df1['Intraday Volume'] = df1['Volume'] - df1['Deliverable Volume']


# In[26]:


df1.head()


# In[27]:


piechart_vars = ['Deliverable Volume','Intraday Volume'];
piechart_values = [df1['Deliverable Volume'].sum(), df1['Intraday Volume'].sum()]
plt.pie(piechart_values, labels=piechart_vars, autopct="%1.2f%%");
plt.title('Types of volume in hdfc bank ')


# In[28]:


df1.sort_values(by='Trades', ascending=False).head()


# In[29]:


df1.sort_values(by='Trades', ascending=False).head()['High']


# In[ ]:





# In[30]:


ma = df1
ma.info()


# In[31]:


ma.index


# In[32]:


ma['Open'].plot(label="No Moving Avg",figsize=(15,7))
ma['MA50'] = ma['Open'].rolling(50).mean()
ma['MA50'].plot(label='MA50')

ma['MA200'] = ma['Open'].rolling(200).mean()
ma['MA200'].plot(label='MA200')

plt.legend()


# In[33]:


ma['Returns'] = (ma['Close']/ ma['Close'].shift(1)) - 1
ma.head()


# In[34]:


ma['Returns'].hist(bins=50)


# In[35]:


ma['Cumulative Return'] = (1 + ma['Returns']).cumprod()
ma.head()


# In[36]:


ma['Cumulative Return'].plot(label='HDFC BANK', figsize=(15,7))
plt.title("Cumulative Return Vs Time")
plt.legend()


# In[ ]:





# In[37]:


df = df.drop(columns=['Symbol'])
df = df.drop(columns=['Series'])


# In[38]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[39]:


x = df[['Close','Volume','Trades']]


# In[40]:


y = df['Turnover']


# In[41]:


x.head()


# In[42]:


y.head()


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 0)


# In[44]:


lr = LinearRegression()


# In[45]:


lr.fit(X_train, y_train)


# In[46]:


X_train.shape, X_test.shape


# In[47]:


y_train = y_train.values.reshape(-1,1)
y_test = y_test.values.reshape(-1,1)


# In[48]:


y_train.shape, y_test.shape


# In[49]:


lr.score(X_test, y_test)


# In[50]:


y_train_pred = lr.predict(X_train)
y_train_pred


# In[51]:


y_test_pred = lr.predict(X_test)
y_test_pred


# In[52]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[53]:


print("R2Score : " ,r2_score(y_test, y_test_pred)) 
print("mean_absolute_error : ",mean_absolute_error(y_test, y_test_pred))
print("mean_squared_error : " ,mean_squared_error(y_test, y_test_pred)) 
print("Root mean_squared_error : ",np.sqrt(mean_squared_error(y_test, y_test_pred)))


# In[54]:


#Random Forest


# In[55]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test) 


# In[56]:


from sklearn.ensemble import RandomForestRegressor

rf_tree = RandomForestRegressor(random_state=0)
rf_tree.fit(X_train_std,y_train)
rf_tree_y_pred = rf_tree.predict(X_train_std)
print("Accuracy: {}".format(rf_tree.score(X_train_std,y_train)))
print("R squared: {}".format(r2_score(y_true=y_train,y_pred=rf_tree_y_pred)))


# In[ ]:


#time series 


# In[58]:


df_train = df[df.Date < "2018"]
df_valid = df[df.Date >= "2018"]


# In[59]:


df_valid.shape


# In[60]:


get_ipython().system('pip install keras')
import tensorflow
from sklearn.preprocessing import MinMaxScaler


# In[61]:


df_train.head().T


# In[62]:


y_train = df_train["Close"]
scaler=MinMaxScaler(feature_range=(0,1))
y_train1=scaler.fit_transform(np.array(df_train['Close']).reshape(-1,1))
y_trainn = df_train["Close"].to_numpy().reshape(-1,1)


# In[63]:


y_valid = df_valid["Close"]
scaler1=MinMaxScaler(feature_range=(0,1))
y_valid1=scaler1.fit_transform(np.array(df_valid['Close']).reshape(-1,1))
y_validn = df_valid["Close"].to_numpy().reshape(-1,1)


# In[64]:


X_train = df_train.drop(columns=["Close"])
X_train1 = X_train.drop("Date", axis=1)
X_train1 = (X_train1-X_train1.min())/(X_train1.max()-X_train1.min())
X_train2 = pd.concat((X_train1, X_train.Date), 1)


# In[65]:


X_valid = df_valid.drop(columns=["Close"])
X_valid1 = X_valid.drop("Date", axis=1)
X_valid1 = (X_valid1-X_valid1.min())/(X_valid1.max()-X_valid1.min())
X_valid2 = pd.concat((X_valid1, X_valid.Date), 1)


# In[66]:


X_train3 = X_train2.drop(columns=['Date']).to_numpy()
X_valid3 = X_valid2.drop(columns=['Date']).to_numpy()


# In[67]:


y_train1 = y_train1.reshape(-1,1)
y_valid1 = y_valid1.reshape(-1,1)


# In[68]:


import xgboost as xgb
import sklearn
from xgboost import XGBRegressor
from xgboost import plot_importance


# In[69]:


xgb = XGBRegressor(n_estimators=1000,learning_rate=0.01)
xgb

xgb.fit(X_train3,y_train1,eval_set=[(X_train3,y_train1),(X_valid3,y_valid1)],early_stopping_rounds=100,verbose=True) # Change verbose to True if you want to see it train


# In[ ]:




