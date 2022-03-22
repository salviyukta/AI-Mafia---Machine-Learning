#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


# In[2]:


dfx = pd.read_csv("C:/Users/Yukta/Downloads/ai_ml/week3/Diabetes_XTrain.csv")
dfy = pd.read_csv("C:/Users/Yukta/Downloads/ai_ml/week3/Diabetes_YTrain.csv")


# In[3]:


print(dfx.shape)


# In[4]:


print(dfx.columns)


# In[5]:


print(dfy.shape)


# In[6]:


print(dfy.columns)


# In[7]:


xtest = pd.read_csv("C:/Users/Yukta/Downloads/ai_ml/week3/Diabetes_Xtest.csv")
test = xtest.values


# In[8]:


x = dfx.values
y = dfy.values


# In[9]:


def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

# Test Time 
def knn(x,y,queryPoint,k=5):
    
    vals = []
    m = x.shape[0]
    
    for i in range(m):
        d = dist(queryPoint,x[i])
        vals.append((d,y[i]))
        
    
    vals = sorted(vals)
    # Nearest/First K points
    vals = vals[:k]
    
    vals = np.array(vals)
    
    #print(vals)
    
    new_vals = np.unique(vals[:,1],return_counts=True)
    print(new_vals)
    
    index = new_vals[1].argmax()
    pred = new_vals[0][index]
    
    return pred
    


# In[24]:


test.shape[0]


# In[54]:


i=0
predictions = []
answers = []
for i in  range(test.shape[0]):
    pred = int(knn(x,y,test[i]))
    print(pred)
    predictions.append(pred)
    answers.append((pred))


# In[55]:


print(predictions[0])
print(predictions[1])


# In[56]:


df = pd.DataFrame(predictions)


# In[57]:


df.head()


# In[58]:


df.to_csv("data.csv")


# In[61]:


data = pd.concat((dfx, dfy), axis = 1)


# In[63]:


data['Outcome'].value_counts().plot.bar()


# In[ ]:




