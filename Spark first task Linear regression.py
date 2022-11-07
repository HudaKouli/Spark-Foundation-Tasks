#!/usr/bin/env python
# coding: utf-8

# # Predicting using supervised ML
# 
# In this section we will see how python Scikit Learn library is used to predict the percentage of an student based on the number of study hours
# (This is simple Linear Regression task as it involves only 2 variables)
# 
# Author Name:Huda kouli
# 

# In[30]:


#importing required libraries 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import median_absolute_error


# In[31]:


#Reading data set and X,y
dataset = pd.read_csv('Data.csv')
print(dataset.head(10))
X = dataset.iloc[:,:1] 
y = dataset.iloc[:, -1]


# In[32]:


#spliting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[33]:


#Training Model
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[34]:


regressor.score(X_train, y_train)


# In[35]:


regressor.score(X_test, y_test)


# In[36]:


#Testing model
y_pred = regressor.predict(X_test)
y_pred 


# In[37]:


#printing y_test to compare the actual output with the predicted values
y_test


# In[38]:


#printing predicted score for one value (hours)
y_pred_new=regressor.predict([[9.25]])
y_pred_new


# In[39]:


mean_absolute_error(y_test, y_pred)


# In[40]:


mean_squared_error(y_test, y_pred)


# In[41]:


median_absolute_error(y_test, y_pred)


# In[42]:


# Visualising the Training,testing sets results
plt.scatter(X_train, y_train, color = 'red')
plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Score for student')
plt.xlabel('Hours')
plt.ylabel('Score')
plt.show()


# In[ ]:





# In[ ]:




