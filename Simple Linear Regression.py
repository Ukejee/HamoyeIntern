#!/usr/bin/env python
# coding: utf-8

# In[82]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import numpy as np

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 30)

test_data_set = pd.read_csv("file:///home/ukeje/Documents/Programming/Hamoye%20Data%20Science%20Stuff/energydata_complete.csv")
test_data_set.head()
independent  = test_data_set.drop(columns=['date','lights'])
scaler = MinMaxScaler()
scaler.fit(independent)
normalized_data = scaler.fit_transform(independent)
processed_data = pd.DataFrame(normalized_data, columns = independent.columns)
processed_data.head()

#From question 12 to 16
x = processed_data.iloc[:, 3:4]
y = processed_data.iloc[:, 11:12]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size = 0.3, random_state = 42)

reg = LinearRegression()
reg.fit(x_train, y_train)


y_pred = reg.predict(x_test)

r2_score = r2_score(y_test, y_pred)
round(r2_score, 2)

mae = mean_absolute_error(y_test, y_pred)
round(mae, 2)

rss = np.sum(np.square(y_test - y_pred))
round(rss, 2)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
round(rmse, 3)

#Question 18
ridge = Ridge(alpha=0.4)
ridge.fit(x_train, y_train)

y_pred = reg.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
round(rmse, 3)


#Question 17
# x = processed_data.drop(columns=['T6'])
# y = processed_data.iloc[:, 11:12]

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size = 0.3, random_state = 42)

# reg = LinearRegression()
# reg.fit(x_train, y_train)

# weight = pd.DataFrame(reg.coef_, index= ['Feature Weight']).transpose()
# attr = pd.DataFrame(x.columns)
# weight_table = pd.concat([weight, attr], axis=1, join="inner")
# print(weight_table["Feature Weight"].min())
# print(weight_table["Feature Weight"].max())
# weight_table

#Question 19 & 20
x = processed_data.drop(columns=['T6'])
y = processed_data.iloc[:, 11:12]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size = 0.3, random_state = 42)

lasso = Lasso(alpha=0.001)
lasso.fit(x_train, y_train)

weight = pd.DataFrame(lasso.coef_).transpose()
attr = pd.DataFrame(x.columns)
weight_table = pd.concat([weight, attr], axis=1, join="inner")
weight

y_pred = lasso.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
round(rmse, 3)


