#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


# In[2]:


# only want first 15 columns from arrhythmia data set
heart = pd.read_csv('arrhythmia.data', header=None)
heart = heart[heart.columns[:15]]


# In[3]:


# got column names from .names file:
heart.columns = ["age", "sex", "height", "weight", "QRS_duration", "PR_interval", "Q-T_interval", "T_interval", "P_interval", "QRS", "T", "P", "QRST", "J", "heart_rate"]

heart.head(10)

# source: https://www.geeksforgeeks.org/add-column-names-to-dataframe-in-pandas/


# In[4]:


# Want to use only variables that are on the same scale. Will look at durations and intervals, which are all in msec.
df = heart[['QRS_duration','PR_interval', 'Q-T_interval', 'T_interval', 'P_interval']]

# drop observations w missing data

df = df.replace({'?':np.nan}).dropna()

# source: https://stackoverflow.com/questions/46269915/delete-all-rows-from-a-dataframe-containing-question-marks

df.head(10)


# In[5]:


# look at all pairwise relationships among the variables in the data set. No super obvious clusters...
ax = pd.plotting.scatter_matrix(df, figsize=(10,8))
# None really look like multiple groups...


# In[7]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# per elbow method, looks like 3 clusters - very faint inflection point in the line.
# I tried range(1,20) to see if I would end up with the 16 categories in the data set, but that did not pan out.


# #### Trying to find a good plot...

# In[11]:


kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(df)
plt.scatter((df["QRS_duration"]), (df["T_interval"]))
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red')
plt.show()


# In[12]:


kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(df)
plt.scatter((df["P_interval"]), (df["T_interval"]))
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red')
plt.show()


# In[13]:


kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(df)
plt.scatter((df["QRS_duration"]), (df["Q-T_interval"]))
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red')
plt.show()


# In[15]:


#Still not great but the best one I've found!
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(df)
plt.scatter((df["QRS_duration"]), (df["P_interval"]))
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red')
plt.show()


# In[ ]:




