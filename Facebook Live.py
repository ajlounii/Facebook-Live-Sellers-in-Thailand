#!/usr/bin/env python
# coding: utf-8

# # Library imports and data analysis

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for statistical data visualization
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import warnings

warnings.filterwarnings('ignore')


# In[3]:


data = 'Live.csv'

df = pd.read_csv(data)


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df.drop(['Column1', 'Column2', 'Column3', 'Column4'], axis=1, inplace=True)


# In[9]:


df.info()


# In[10]:


df.describe()


# In[11]:


df['status_id'].unique()


# In[12]:


len(df['status_id'].unique())


# In[13]:


df['status_published'].unique()


# In[14]:


len(df['status_published'].unique())


# In[15]:


df['status_type'].unique()


# In[16]:


len(df['status_type'].unique())


# In[17]:


df.drop(['status_id', 'status_published'], axis=1, inplace=True)


# In[18]:


df.info()


# In[19]:


df.head()


# # Declare feature vector and target variable

# In[20]:


X = df

y = df['status_type']


# # Converting Status_Type into integers

# In[21]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X['status_type'] = le.fit_transform(X['status_type'])

y = le.transform(y)


# # X Analysis

# In[22]:


X.info()


# In[23]:


X.head()


# # Scaling

# In[24]:


cols = X.columns


# In[25]:


from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()

X = ms.fit_transform(X)


# In[26]:


X = pd.DataFrame(X, columns=[cols])


# In[27]:


X.head()


# # 2 clusters

# In[28]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0) 

kmeans.fit(X)


# In[29]:


kmeans.cluster_centers_


# # Inertia Analysis

# In[30]:


kmeans.inertia_


# # Silhouette Analysis

# In[31]:


from sklearn.metrics import silhouette_score

# Silhouette analysis
range_n_clusters = [2, 3, 4, 5, 6]

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(X)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))


# # elbow method

# In[32]:


from sklearn.cluster import KMeans
cs = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    cs.append(kmeans.inertia_)
plt.plot(range(1, 11), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()


# # Two Clusters Accuracy

# In[33]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2,random_state=0)

#clusters=kmeans.fit(X)
label=kmeans.fit_predict(X)

labels = kmeans.labels_

# check how many of the samples were correctly labeled

correct_labels = sum(y == labels)

print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))

print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# # 3 Clusters

# In[34]:


kmeans = KMeans(n_clusters=3, random_state=0)

#kmeans.fit(X)
label=kmeans.fit_predict(X)
# check how many of the samples were correctly labeled
labels = kmeans.labels_

correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# # 4 Clusters

# In[35]:


kmeans = KMeans(n_clusters=4, random_state=0)

#kmeans.fit(X)
label=kmeans.fit_predict(X)
# check how many of the samples were correctly labeled
labels = kmeans.labels_

correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# # 5 Clusters

# In[36]:


kmeans = KMeans(n_clusters=5, random_state=0)

kmeans.fit(X)

# check how many of the samples were correctly labeled
labels = kmeans.labels_

correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# # 6 Clusters

# In[37]:


kmeans = KMeans(n_clusters=6, random_state=0)

kmeans.fit(X)

# check how many of the samples were correctly labeled
labels = kmeans.labels_

correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# # Spectral Clustering, 2 Clusters

# In[38]:


from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import SpectralClustering
# define the model
model = SpectralClustering(n_clusters=2)
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
#for cluster in clusters:
	# get row indexes for samples with this cluster
#	row_ix = where(yhat == cluster)
#	# create scatter of these samples
#	plt.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
#plt.show()


# In[39]:


labels = model.labels_

correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# # 3 Clusters

# In[40]:


from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import SpectralClustering
# define the model
model = SpectralClustering(n_clusters=3)
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
#for cluster in clusters:
	# get row indexes for samples with this cluster
#	row_ix = where(yhat == cluster)
#	# create scatter of these samples
#	plt.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
#plt.show()


# In[41]:


labels = model.labels_

correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# # 4 Clusters

# In[42]:


from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import SpectralClustering
# define the model
model = SpectralClustering(n_clusters=4)
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
#for cluster in clusters:
	# get row indexes for samples with this cluster
#	row_ix = where(yhat == cluster)
#	# create scatter of these samples
#	plt.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
#plt.show()


# In[43]:


labels = model.labels_

correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# In[ ]:




