#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import nltk 
import string
import re
get_ipython().run_line_magic('matplotlib', 'inline')


# In[23]:


tweet_data=pd.read_csv('test_file.csv')
tweet_data.head()


# In[24]:


tweet_data.columns


# In[25]:


tweet_data = tweet_data[['text']]


# In[26]:


tweet_data


# In[27]:


def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text) #removes numbers from text
    return text


tweet_data['clean_text']=tweet_data['text'].apply(lambda x: remove_punct(x))
tweet_data.head(5)


# In[29]:


def tokenization(text):
    text = re.split('\W+', text) #splitting each sentence/ tweet into its individual words
    return text

tweet_data['Tweet_tokenized'] = tweet_data['clean_text'].apply(lambda x: tokenization(x.lower()))
tweet_data.head()


# In[30]:


nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
stopwords


# In[31]:


def remove_stopwords(text):
    text = [word for word in text if word not in stopwords]
    return text
    
tweet_data['Tweet_without_stop'] = tweet_data['Tweet_tokenized'].apply(lambda x: remove_stopwords(x))
tweet_data.head(5)


# In[32]:


ps = nltk.PorterStemmer()

def stemming(text):
    text = [ps.stem(word) for word in text]
    return text

tweet_data['Tweet_stemmed'] = tweet_data['Tweet_without_stop'].apply(lambda x: stemming(x))
tweet_data.head()


# In[34]:


nltk.download('wordnet')
wordnet = nltk.WordNetLemmatizer()

def lemmatizer(text):
    text = [wordnet.lemmatize(word) for word in text]
    return text

tweet_data['Tweet_lemmatized'] = tweet_data['Tweet_without_stop'].apply(lambda x: lemmatizer(x))
tweet_data.head()


# In[ ]:


#Count Vectorizer


# In[35]:


from random import sample
clean_tweets=list(set(tweet_data['clean_text']))
sample=sample(clean_tweets,20)


# In[36]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sample)
vectorizer.get_feature_names()


# In[37]:


count_vect_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
count_vect_df.head()


# In[ ]:


#Data Visualization


# In[38]:


all_clean_words=[]

for i in tweet_data['Tweet_lemmatized']:
    for j in i:
        all_clean_words.append(j)

all_clean_words=list(set(all_clean_words)) #removes duplicate values from the list

clean_words_str=' '.join(all_clean_words)


# In[39]:


from wordcloud import WordCloud
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(clean_words_str)
 
# plot the WordCloud image                      
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()


# In[43]:


tweet_data.to_csv('Clean_Twitter_data.csv')

