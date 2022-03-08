#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1. Import the numpy package under the name as np
import numpy as np


# In[2]:


#2. Print the numpy version and the configuration
print(np.__version__)
print(np.show_config())


# In[3]:


#3. Create a null vector of size 10
x = np.zeros(10)


# In[5]:


print(x)


# In[6]:


print(type(x))


# In[8]:


#4. How to find the memory size of any array
#memory size = number of items * size of each item
print(x.size)
print(x.itemsize)
print(x.itemsize * x.size)


# In[9]:


#5. How to get the documentation of the numpy add function from the command line?
get_ipython().run_line_magic('pinfo', 'np.add')


# In[10]:


#6. Create a null vector of size 10 but the fifth value which is 1
y = np.zeros(10)
y[4] = 1
print(y)


# In[11]:


#7. Create a vector with values ranging from 10 to 49
z = np.arange(10,50)
print(z)


# In[12]:


#8. Reverse a vector (first element becomes last)
z = z[::-1]
print(z)


# In[16]:


#9. Create a 3x3 matrix with values ranging from 0 to 8
a = np.arange(9).reshape(3,3)
print(a)


# In[27]:


#10. Find indices of non-zero elements from [1,2,0,0,4,0]
p = np.array([1,2,0,0,4,0])
q = np.nonzero(p)
print(q)


# In[29]:


#11. Create a 3x3 identity matrix
r = np.eye(3,3)
print(r)


# In[34]:


#12. Create a 3x3x3 array with random values
n = np.random.random((3,3,3))
print(n)


# In[37]:


#13. Create a 10x10 array with random values and find the minimum and maximum values
n = np.random.random((10,10))
nmin = n.min()
nmax = n.max()
print(n)
print(nmin,nmax,sep="\n")


# In[38]:


#14. Create a random vector of size 30 and find the mean value
rand = np.random.random(30)
print(rand)
print(rand.mean())


# In[44]:


#15. Create a 2d array with 1 on the border and 0 inside
nz = np.zeros((10,10))
nz = np.pad(nz,pad_width=1,mode='constant',constant_values=1)
print(nz)


# In[50]:


#16. How to add a border (filled with 0's) around an existing array?
nz = np.ones((10,10))
nz = np.pad(nz,pad_width=1,mode='constant',constant_values=0)
print(nz)


# In[52]:


#17. What is the result of the following expression?
'''0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
np.nan in set([np.nan])
0.3 == 3 * 0.1'''

0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
np.nan in set([np.nan])
0.3 == 3 * 0.1


# In[63]:


#18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal
Z=np.diag(1+np.arange(4),k=-1)
print(Z)


# In[64]:


#19. Create a 8x8 matrix and fill it with a checkerboard pattern
result = np.zeros((8,8))
result[::2,1::2]=1
result[1::2,::2]=1
print(result)


# In[65]:


#20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?
print(np.unravel_index(100,(6,7,8)))


# In[ ]:




