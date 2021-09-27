import numpy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.random import normal
from sklearn.mixture import GaussianMixture
import scipy.stats
from sklearn import mixture
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


model = GaussianMixture(n_components=2, init_params='random')
ymod = model.fit(X)
yhat = model.predict(X)
print(yhat[:100])
print(yhat[-100:])
plt.scatter(X[:,0], X[:, 1], s = 1)

centers = np.empty(shape=(ymod.n_components, X.shape[1]))
for i in range(ymod.n_components):
    density = scipy.stats.multivariate_normal(cov=ymod.covariances_[i], mean=ymod.means_[i]).logpdf(X)
    centers[i, :] = X[np.argmax(density)]
plt.scatter(centers[:, 0], centers[:, 1], s=20)
plt.show()
