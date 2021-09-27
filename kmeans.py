import numpy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import csv
from sklearn.preprocessing import LabelEncoder

col_list= ["REQUEST_ORIGIN", "NEIGHBORHOOD","X","Y"]
df = pd.read_csv("test.csv", usecols=col_list)
df=df[:200]  
print(df.head())
#data = pd.read_excel("test.xls") 
p=df['REQUEST_ORIGIN'].tolist()
q=df['NEIGHBORHOOD'].tolist()
r=df['X'].tolist()
s=df['Y'].tolist()

a = LabelEncoder()
b = LabelEncoder()
c = LabelEncoder()

a=a.fit_transform(df['REQUEST_ORIGIN'])
b=b.fit_transform(df['X'])
c=c.fit_transform(df['Y'])

Xp=np.array(a)
Xq=np.array(b)
Xr=np.array(c)


#plot for all depenencies
plt.scatter(Xq, Xp, c='black', s=7)
plt.scatter(Xq, Xr, c='black', s=7)

X = np.array(list(zip(Xp,Xq,Xr)))
plt.show()


# Number of clusters
k = 3
# X coordinates of random centroids
C_x = np.random.uniform(0, np.max(X)-0.2, size=k)
# Y coordinates of random centroids
C_y = np.random.uniform(0, np.max(X)-0.2, size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float)
print("The starting set of centroids is as below: ")
print(C)

#plot for the centroids
plt.scatter(Xq, Xp, c='black', s=7)
plt.scatter(Xq, Xr, c='blue', s=7)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')
plt.show()

#clustering to new values

kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=200, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)
plt.scatter(X[:,0], X[:,1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='blue')
plt.scatter(X[:,1], X[:,2])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='green')
plt.show()