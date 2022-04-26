#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gensim
from gensim.models import word2vec
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz',binary=True)


# In[3]:


import numpy as np


# In[4]:


def odd_one_out(words):
    """Accepts a list of words and returns the odd word"""
    
    # Generate all word embeddings for the given list
    all_word_vectors = [word_vectors[w] for w in words]
    avg_vector = np.mean(all_word_vectors,axis=0)
    print(avg_vector.shape)
    
    #Iterate over every word and find similarity
    odd_one_out = None
    min_similarity = 1.0 #Very high value
    
    for w in words:
        sim = cosine_similarity([word_vectors[w]],[avg_vector])
        if sim < min_similarity:
            min_similarity = sim
            odd_one_out = w
    
        print("Similairy btw %s and avg vector is %.2f"%(w,sim))
            
    return odd_one_out


# In[7]:


import pandas as pd


# In[36]:


test_data = pd.read_csv("C:/Users/Yukta/Downloads/ai_ml/week5/word2vec_assignment/Test_data.csv")


# In[37]:


print(test_data)


# In[38]:


print(test_data.shape)


# In[39]:


test_data = test_data.values


# In[40]:


print(test_data)


# In[42]:


pred = []
for i in range(test_data.shape[0]):
    p = odd_one_out(test_data[i])
    print(p)
    pred.append(p)


# In[45]:


result = pd.DataFrame(pred)
print(result)


# In[44]:


result.to_csv("one_one_out_result.csv", index = False)


# In[ ]:




