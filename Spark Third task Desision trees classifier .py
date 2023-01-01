#!/usr/bin/env python
# coding: utf-8

# # Predicting Using Desision Tree Classifier
# In this section we will see how python Scikit Learn library is used to predict the right class accordingly
# 
# Author Name: Huda Kouli

# In[1]:


#importing required libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[3]:


#Reading data set and X,y
dataset = pd.read_csv('Iris.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values


# In[4]:


#spliting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[5]:


#Training Model
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[6]:


classifier.score(X_train, y_train)


# In[7]:


print(classifier.score(X_test , y_test))


# In[8]:


#Testing model
y_pred = classifier.predict(X_test)
y_pred 


# In[9]:


#printing y_test to compare the actual output with the predicted values
y_test


# In[10]:


#printing predicted score for new value
y_pred2=classifier.predict(np.array([5,4.8,3.3,1.4,0.2]).reshape(1,-1))
y_pred2


# In[11]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[12]:


import seaborn as sns
sns.heatmap(cm, center=True)
plt.show()


# In[13]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[ ]:




