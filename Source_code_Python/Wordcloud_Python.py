#!/usr/bin/env python
# coding: utf-8

# In[118]:


#Loading the required libaries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
import os
import re
import warnings
from sys import exit
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
#from nltk.corpus import stopwords
from sklearn.cluster import KMeans


# In[84]:


df = pd.read_csv("Clean_Twitter_data.csv")
twitter_data = df[['Tweet_lemmatized']]


# In[85]:


twitter_data


# In[86]:


stopwords = set(STOPWORDS)
letters_only = re.sub("[^a-zA-Z]+",  # Search for all non-letters

                         " ",          # Replace all non-letters with spaces

str(twitter_data['Tweet_lemmatized']))

wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(letters_only)


# In[87]:


plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()


# In[96]:


#Comparing Countvectorizer and Tfidf Vectorizer


# In[106]:


#CountVectorizer
CV=CountVectorizer(input='content',
                      stop_words='english',
                      #decode_error='ignore'
                      )
The_DTM_CV=CV.fit_transform(twitter_data['Tweet_lemmatized'])
TheColumnNames_CV=CV.get_feature_names()


# In[111]:


#regex module
import re

pattern = r'[0-9]'

# Match all digits in the string and replace them with an empty string
New_TheColumnNames = re.sub(pattern, '', str(TheColumnNames_CV))

#print(New_TheColumnNames)


# In[112]:


#print("The vocabulary is:",TheColumnNames,"\n\n")
#The second step is to use pandas to create data frames
The_DF=pd.DataFrame(The_DTM.toarray(),columns=New_TheColumnNames)


# In[113]:


The_DF = The_DF.filter(regex='^\D')


# In[114]:


The_DF.to_csv("DTM_CV.csv")


# In[102]:


#TfidfVectorizer
TF = TfidfVectorizer(stop_words='english')
The_DTM_TFidf = TF.fit_transform(twitter_data['Tweet_lemmatized'])
TheColumnNames_TF=TF.get_feature_names()


# In[98]:


#New_TheColumnNames_2 = re.sub(pattern, '', str(TheColumnNames_TF))


# In[103]:


The_DF_TF=pd.DataFrame(The_DTM_TFidf.toarray(),columns=TheColumnNames)


# In[104]:


The_DF_TF = The_DF_TF.filter(regex='^\D')


# In[105]:


The_DF_TF.to_csv("DTM_TF.csv")


# In[ ]:





# In[140]:


#Kmeans for K = 3
km = KMeans(n_clusters=3)
km.fit(The_DTM_CV)


# In[141]:


pca = PCA(n_components=2, random_state=2)
reduced_features = pca.fit_transform(The_DTM_CV.toarray())

# reduce the cluster centers to 2D\
reduced_cluster_centers = pca.transform(km.cluster_centers_)


# In[143]:


from matplotlib.pyplot import figure

figure(figsize=(10, 12), dpi=80)
plt.scatter(reduced_features[:,0], reduced_features[:,1], c=km.predict(The_DTM_CV))
plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='r')


# In[134]:


#Kmeans for K = 5
km = KMeans(n_clusters=5)
km.fit(The_DTM_CV)


# In[135]:


pca = PCA(n_components=2, random_state=2)
reduced_features = pca.fit_transform(The_DTM_CV.toarray())

# reduce the cluster centers to 2D\
reduced_cluster_centers = pca.transform(km.cluster_centers_)


# In[144]:


from matplotlib.pyplot import figure

figure(figsize=(10, 12), dpi=80)
plt.scatter(reduced_features[:,0], reduced_features[:,1], c=km.predict(The_DTM_CV))
plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='r')


# In[137]:


#Kmeans for K = 8
km = KMeans(n_clusters=8)
km.fit(The_DTM_CV)


# In[138]:


pca = PCA(n_components=2, random_state=2)
reduced_features = pca.fit_transform(The_DTM_CV.toarray())

# reduce the cluster centers to 2D\
reduced_cluster_centers = pca.transform(km.cluster_centers_)


# In[145]:


from matplotlib.pyplot import figure

figure(figsize=(10, 12), dpi=80)
plt.scatter(reduced_features[:,0], reduced_features[:,1], c=km.predict(The_DTM_CV))
plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='r')

