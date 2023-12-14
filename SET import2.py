#!/usr/bin/env python
# coding: utf-8

# In[6]:


from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse
import pandas as pd


# In[4]:


df = pd.read_csv('C:/Users/KAMDEO SINGH/OneDrive/Desktop/new import (1).csv',index_col='date')
df.index.freq = 'MS'
df


# In[3]:


df.isna().sum()


# In[68]:


df.head(13)


# In[69]:


df.tail(11)


# In[71]:


df.imports = pd.to_numeric(import1.imports, errors='coerce')


# In[13]:


import1.reset_index(inplace=True)


# In[72]:


df.plot(figsize=(15,5))


# In[73]:


import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(10,6)})
plt.plot(df['imports'])


# In[1]:


import1.isna().sum()


# In[76]:


# Multiplicative Decomposition
mul_result = seasonal_decompose(df['imports'],model='multiplicative',period=1)

#Additive Decomposition
add_result = seasonal_decompose(df['imports'],model='additive',period=1)


# In[75]:


#plot
plt.rcParams.update({'figure.figsize':(10,10)})
mul_result.plot().suptitle('\nMultiplicatice Decompose',fontsize=12)


# In[77]:


add_result.plot().suptitle('\nAdditive Decompose',fontsize=12)
plt.show()


# In[78]:


#Additive
new_df_add = pd.concat([add_result.seasonal, add_result.trend, add_result.resid, add_result.observed], axis=1)
new_df_add.columns = ['seasonality','trend','residual','actual_import1']
new_df_add.head(5)


# In[79]:


#Multiplicative
new_df_mult = pd.concat([mul_result.seasonal, mul_result.trend, mul_result.resid, mul_result.observed], axis=1)
new_df_mult.columns = ['seasonality','trend','residual','actual_import1']
new_df_mult.head(5)


# In[50]:


#ADFULLER TEST FOR STATIONARITY
from statsmodels.tsa.stattools import adfuller


# In[80]:


#ADF Test null hypothesis non stationarity

adfuller_result = adfuller(df.imports,autolag='AIC')

print(f'ADF statistic: {adfuller_result[0]}')
print(f'p-value: {adfuller_result[1]}')

for key, value in adfuller_result[4].items():
    print('Critical values:')
    print(f'    {key},  {value}')


# In[89]:


df_diff = df.imports.diff()[1:]


# In[90]:


df_diff = pd.DataFrame(df_diff)


# In[91]:


#ADF Test nul hypothesis non stationarity

adfuller_result = adfuller(df_diff.imports,autolag='AIC')

print(f'ADF statistic: {adfuller_result[0]}')
print(f'p-value: {adfuller_result[1]}')

for key, value in adfuller_result[4].items():
    print('Critical values:')
    print(f'    {key},  {value}')


# In[92]:


df_diff


# In[93]:


df_diff2=df_diff.copy()
df_diff2


# In[94]:


df_diff.plot(figsize=(15,5))


# In[95]:


df_diff1=df_diff


# In[96]:


df_diff1['imports_LastMonth']=df_diff1['imports'].shift(+1)
df_diff1['imports_2Monthsback']=df_diff1['imports'].shift(+2)
df_diff1['imports_3Monthsback']=df_diff1['imports'].shift(+3)
df_diff1


# In[97]:


df_diff2


# In[98]:


df_diff1=df_diff1.dropna()
df_diff1


# In[99]:


from sklearn.linear_model import LinearRegression
lin_model=LinearRegression()


# In[100]:


from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(n_estimators=100,max_features=3, random_state=1)


# In[101]:


import numpy as np
x1,x2,x3,y=df_diff1['imports_LastMonth'],df_diff1['imports_2Monthsback'],df_diff1['imports_3Monthsback'],df_diff1['imports']
x1,x2,x3,y=np.array(x1),np.array(x2),np.array(x3),np.array(y)
x1,x2,x3,y=x1.reshape(-1,1),x2.reshape(-1,1),x3.reshape(-1,1),y.reshape(-1,1)
final_x=np.concatenate((x1,x2,x3),axis=1)
print(final_x)


# In[102]:


X_train,X_test,y_train,y_test=final_x[:-30],final_x[-30:],y[:-30],y[-30:]


# In[103]:


model.fit(X_train,y_train)
lin_model.fit(X_train,y_train)


# In[104]:


lin_model.score(X_train,y_train)


# In[105]:


model.score(X_train,y_train)


# In[106]:


pred=model.predict(X_test)
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,8)
plt.plot(pred,label='Random_Forest_Predictions')
plt.plot(y_test,label='Actual imports')
plt.legend(loc="upper left")
plt.show()


# In[107]:


lin_pred=lin_model.predict(X_test)
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11,6)
plt.plot(lin_pred,label='Linear_Regression_Predictions')
plt.plot(y_test,label='Actual imports')
plt.legend(loc="upper left")
plt.show()


# In[108]:


from sklearn.metrics import mean_squared_error
from math import sqrt
rmse_rf=sqrt(mean_squared_error(pred,y_test))
rmse_lr=sqrt(mean_squared_error(lin_pred,y_test))


# In[109]:


print('Mean Squared Error for Random Forest Model is:',rmse_rf)
print('Mean Squared Error for Linear Regression Model is:',rmse_lr)


# In[113]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score


# In[114]:


# k nearest neighbours regression model with accuracy and MSE calculation
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
knn_r2 = r2_score(y_test, knn_predictions)
knn_mse = mean_squared_error(y_test, knn_predictions)
print("K-Nearest Neighbors Regression R-squared:", knn_r2)
print("K-Nearest Neighbors Regression MSE:", knn_mse)


# In[116]:


from sklearn.tree import DecisionTreeRegressor


# In[117]:


# Decision tree regression model
dt_reg = DecisionTreeRegressor(random_state=42)
dt_reg.fit(X_train, y_train)
y_pred_dt = dt_reg.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
print('Decision tree regression MSE:', mse_dt)
print('Decision tree regression accuracy:', r2_dt)


# In[118]:


from sklearn.linear_model import Ridge


# In[119]:


# Ridge regression model with accuracy and MSE calculation
ridge_model = Ridge(alpha=0.5)
ridge_model.fit(X_train, y_train)
ridge_predictions = ridge_model.predict(X_test)
ridge_r2 = r2_score(y_test, ridge_predictions)
ridge_mse = mean_squared_error(y_test, ridge_predictions)
print("Ridge Regression R-squared:", ridge_r2)
print("Ridge Regression MSE:", ridge_mse)


# In[120]:


# Linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)
print('Linear regression MSE:', mse_lin)
print('Linear regression accuracy:', r2_lin)


# In[1]:


import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


# In[2]:


#create data with linear trend
np.random.seed(123)
t = np.arange(100)
y = t + 2 * np.random.normal(size = 100)#linear trend


# In[3]:


t_train = t[:50].reshape(-1,1)
t_test = t[50:].reshape(-1,1)


# In[5]:


y_train1 = y[:50]
y_test1 = y[50:]


# In[6]:


tree = DecisionTreeRegressor(max_depth = 2)
tree.fit(t_train, y_train)


# In[7]:


y_pred_train = tree.predict(t_train)
y_pred_test = tree.predict(t_test)


# In[8]:


plt.figure(figsize = (16,8))
plt.plot(t_train.reshape(-1), y_train, label = "Training data", color="blue", lw=2)
plt.plot(np.concatenate([np.array(t_train[-1]),t_test.reshape(-1)]), np.concatenate([[y_train[-1]],y_test]), label = "Test data", color="blue", ls = "dotted", lw=2)


# In[ ]:




