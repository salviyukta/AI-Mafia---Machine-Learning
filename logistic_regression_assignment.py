#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[6]:


x = pd.read_csv("C:/Users/Yukta/Downloads/ai_ml/week2/assignment3/Logistic_X_Train.csv")
y = pd.read_csv("C:/Users/Yukta/Downloads/ai_ml/week2/assignment3/Logistic_Y_Train.csv")
x_test = pd.read_csv("C:/Users/Yukta/Downloads/ai_ml/week2/assignment3/Logistic_X_Test.csv")


# In[7]:


data = pd.concat((x, y), axis = 1)


# In[8]:


features = data[ ['f1', 'f2', 'f3']]
target = data['label']


# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size = 0.25)


# In[10]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[11]:


model.fit(X_train, y_train)


# In[13]:


print(model.coef_)
print(model.intercept_)


# In[14]:


print("Training Score", model.score(X_train, y_train))


# In[15]:


print("Validation Score", model.score(X_val, y_val))


# In[17]:


y_pred = model.predict(x_test)


# In[18]:


r = pd.DataFrame(y_pred, columns = ['result'])


# In[20]:


r.to_csv("chemical_result.csv", index = False)


# In[ ]:




