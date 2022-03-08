#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import pandas
import pandas as pd


# In[7]:


# load imdb dataset as pandas dataframe
imdb_df = pd.read_csv("C:/Users/Yukta/Downloads/ai_ml/week1/pandas/imdb_1000.csv")


# In[12]:


# show first 5 rows of imdb_df
imdb_df.head()


# In[94]:


# load bikes dataset as pandas dataframe
bikes_df = pd.read_csv("C:/Users/Yukta/Downloads/ai_ml/week1/pandas/bikes.csv",sep=";", parse_dates=['Date'],dayfirst=True,index_col='Date')


# In[27]:


# show first 3 rows of bikes_df
bikes_df.head(n=3)


# In[29]:


# list columns of imdb_df
imdb_df.columns


# In[33]:


# what are the datatypes of values in columns
imdb_df.dtypes


# In[35]:


# list first 5 movie titles
imdb_df['title'].head()


# In[37]:


# show only movie title and genre
imdb_df[['title','genre']]


# In[41]:


# show the type of duration column
imdb_df.duration.dtype


# In[45]:


# show duration values of movies as numpy arrays
imdb_df.duration.values


# In[50]:


# convert all the movie titles to uppercase
uppercase = lambda x : x.upper()
imdb_df['title'].apply(uppercase)


# In[58]:


# plot the bikers travelling to Berri1 over the year
import matplotlib.pyplot as plt
#%matplotlib inline
bikes_df['Berri1'].plot()


# In[61]:


# plot all the columns of bikes_df
bikes_df.plot()


# In[64]:


# what are the unique genre in imdb_df?
imdb_df['genre'].value_counts()


# In[65]:


# plotting value counts of unique genres as a bar chart
imdb_df['genre'].value_counts().plot.bar()


# In[69]:


imdb_df['genre'].value_counts().plot.pie(figsize=(10,7))


# In[71]:


# show index of bikes_df
bikes_df.index


# In[73]:


# get row for date 2012-01-01
bikes_df.loc['2012-01-01']


# In[75]:


# show 11th row of imdb_df using iloc
imdb_df.iloc[10]


# In[79]:


# select only those movies where genre is adventure
imdb_df[imdb_df['genre'] =='Adventure']


# In[81]:


# which genre has highest number of movies with star rating above 8 and duration more than 130 minutes?
imdb_df[(imdb_df['star_rating'] > 8) & (imdb_df['duration'] > 130)]


# In[125]:


# add a weekday column to bikes_df
bikes_df['weekday'] = bikes_df.index.weekday


# In[95]:


# remove column 'Unnamed: 1' from bikes_df
bikes_df.drop('Unnamed: 1', axis=1, inplace=True)


# In[96]:


bikes_df.head()


# In[100]:


# remove row no. 1 from bikes_df
#bikes_df.drop(bikes_df.index[0]).head()
bikes_df.drop(bikes_df.index[0], axis=0)


# In[101]:


bikes_df


# In[113]:


# group imdb_df by movie genres
group_genre = imdb_df.groupby('genre')


# In[114]:


# get crime movies group
group_genre.get_group('Crime')


# In[129]:


# get mean of movie durations for each group
group_genre.aggregate('mean')


# In[118]:


# change duration of all movies in a particular genre to mean duration of the group
imdb_df['new_duration'] = group_genre['duration'].transform(lambda x : x.mean())


# In[121]:


# drop groups/genres that do not have average movie duration greater than 120.
new_average = group_genre.filter(lambda x : x['duration'].mean()>120)


# In[126]:


# group weekday wise bikers count
weekday_group = bikes_df.groupby('weekday')


# In[131]:


# get weekday wise biker count
weekday_count = weekday_group.aggregate(sum)


# In[132]:


# set index of the resulting aggregation by weekday names
weekday_count.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


# In[133]:


# plot weekday wise biker count for 'Berri1'
weekday_count['Berri1']


# In[135]:


weekday_count['Berri1'].plot.bar()


# In[ ]:




