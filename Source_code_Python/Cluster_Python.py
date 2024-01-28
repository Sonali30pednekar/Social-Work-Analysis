#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import preprocessing
import pylab as pl
from sklearn import decomposition
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering

import scipy.cluster.hierarchy as hc

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


# In[9]:


filename= pd.read_csv("DTM_CV.csv")


# In[14]:


# KMEANS
# Use k-means clustering on the data.

# Create clusters 
k = 4
## Sklearn required you to instantiate first
kmeans = KMeans(n_clusters=k)
kmeans.fit(filename)   ## run kmeans

labels = kmeans.labels_
print(labels)

centroids = kmeans.cluster_centers_
print(centroids)

prediction = kmeans.predict(filename)
print(prediction)


# In[16]:


#Look at best values for k 
SS_dist = []

values_for_k=range(2,7)
#print(values_for_k)

for k_val in values_for_k:
    print(k_val)
    k_means = KMeans(n_clusters=k_val)
    model = k_means.fit(filename)
    SS_dist.append(k_means.inertia_)
    
print(SS_dist)
print(values_for_k)

plt.plot(values_for_k, SS_dist, 'bx-')
plt.xlabel('value')
plt.ylabel('Sum of squared distances')
plt.title('Elbow method for optimal k Choice')
plt.show()


# In[30]:


# Look at Silhouette

Sih=[]
Cal=[]
k_range=range(2,20)

for k in k_range:
    k_means_n = KMeans(n_clusters=k)
    model = k_means_n.fit(filename)
    Pred = k_means_n.predict(filename)
    labels_n = k_means_n.labels_
    R1=metrics.silhouette_score(filename, labels_n, metric = 'euclidean')
    R2=metrics.calinski_harabasz_score(filename, labels_n)
    Sih.append(R1)
    Cal.append(R2)

print(Sih) ## higher is better
print(Cal) ## higher is better

fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
ax1.plot(k_range,Sih)
ax1.set_title("Silhouette")
ax1.set_xlabel("")



ax2.plot(k_range,Cal)
ax2.set_title("Calinski_Harabasz_Score")
ax2.set_xlabel("k values")


# In[19]:


##      Look at Clusters
# It is often best to normalize the data 
## before applying the fit method
## There are many normalization options
## This is an example of using the z score
file_norm =(filename - filename.mean()) / filename.std()
#print(file_norm)


# In[32]:


#PCA - Principle Component Analysis
print(file_norm.shape[0])   ## num rows
print(file_norm.shape[1])   ## num cols

NumCols=file_norm.shape[1]

## Instantiated my own copy of PCA
My_pca = PCA(n_components=4)  ## I want the two prin columns

## Transpose it
file_norm=np.transpose(file_norm)
My_pca.fit(file_norm)

print(My_pca)
print(My_pca.components_.T)

# Reformat and view results
Comps = pd.DataFrame(My_pca.components_.T,
                        columns=['PC%s' % _ for _ in range(4)],
                        index=file_norm.columns
                        )
print(Comps)
print(Comps.iloc[:,0])
#RowNames = list(Comps.index)
#print(RowNames)


# In[26]:


MyDBSCAN = DBSCAN(eps=6, min_samples=2)
## eps:
    ## The maximum distance between two samples for 
    ##one to be considered as in the neighborhood of the other.
MyDBSCAN.fit_predict(filename)
print(MyDBSCAN.labels_)


# In[28]:


##  Hierarchical 

MyHC = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
FIT=MyHC.fit(filename)
HC_labels = MyHC.labels_
print(HC_labels)


plt.figure(figsize =(12, 12))
plt.title('Hierarchical Clustering')
dendro = hc.dendrogram((hc.linkage(filename, method ='ward')))

## WARD
## Recursively merges the pair of clusters that 
## minimally increases within-cluster variance.

from sklearn.metrics.pairwise import euclidean_distances
EDist=euclidean_distances(filename)
print(EDist)

