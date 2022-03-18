#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[24]:


dfx = pd.read_csv("C:/Users/Yukta/Downloads/ai_ml/week2/assignment1/Linear_X_Train.csv")
dfy = pd.read_csv("C:/Users/Yukta/Downloads/ai_ml/week2/assignment1/Linear_Y_Train.csv")

x = dfx.values#to numpy array
y = dfy.values

x = x.reshape((-1,1))#to linear array
y = y.reshape((-1,1))


# In[25]:


plt.scatter(x,y,color='red')


# In[26]:


from sklearn.linear_model import LinearRegression


# In[27]:


model = LinearRegression()


# In[28]:


model.fit(x,y)


# In[29]:


output = model.predict(x)


# In[30]:


bias = model.intercept_
coeff = model.coef_


# In[31]:


plt.scatter(x,y,color='red',label='data')
plt.plot(x,output,color='black',label='prediction')
plt.legend()
plt.show()


# In[32]:


xtest = pd.read_csv("C:/Users/Yukta/Downloads/ai_ml/week2/assignment1/Linear_X_Test.csv")


# In[35]:


ytest = model.predict(xtest) 


# In[36]:


df = pd.DataFrame(data = ytest,columns=["prediction"])
df.to_csv("smart_watch.csv",index=False)


# In[ ]:





# In[ ]:




