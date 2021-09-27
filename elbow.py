import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder

#Here we would load the dataset and print it
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

#plot the graph
plt.scatter(Xq, Xp, c='black', s=7)
plt.scatter(Xq, Xr, c='black', s=7)

X = np.array(list(zip(Xp,Xq,Xr)))
plt.show()


# k means determine k
distortions = []
K = range(1,26)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
