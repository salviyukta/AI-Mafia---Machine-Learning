#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[10]:


data = pd.read_csv("C:/Users/Yukta/Downloads/ai_ml/week2/assignment2/Train.csv")

xtest = pd.read_csv("C:/Users/Yukta/Downloads/ai_ml/week2/assignment2/Test.csv")


# In[11]:


data.head()


# In[12]:


features = data[ ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']]
target = data['target']


# In[13]:


from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(features, target, test_size = 0.25, random_state = 33)


# In[14]:


from sklearn.linear_model import LinearRegression


# In[15]:


model = LinearRegression()


# In[16]:


model.fit(x_train,y_train)


# In[22]:


print("Training Score", model.score(X_train, y_train))


# In[23]:


print("Validation Score", model.score(X_val, y_val))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




