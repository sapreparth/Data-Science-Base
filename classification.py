import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import plot_confusion_matrix
from sklearn import mixture
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

#Here we would load the dataset and print it
col_list= ["POLICE_ZONE", "WARD","X","Y","STATUS"]
df = pd.read_csv("test.csv", usecols=col_list)
df=df[:200]  
print(df.head())

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)
    
clean_dataset(df)
    
#data = pd.read_excel("test.xls") 
p=df['POLICE_ZONE'].tolist()
q=df['WARD'].tolist()
t=df['STATUS'].tolist()

a = LabelEncoder()
b = LabelEncoder()
c = LabelEncoder()

a=a.fit_transform(df['POLICE_ZONE'])
b=b.fit_transform(df['WARD'])
c=c.fit_transform(df['STATUS'])

Xp=np.array(a)
Xq=np.array(b)
Xr=np.array(c)

#plot the graph
plt.scatter(Xq, Xp, c='black', s=7)
plt.scatter(Xq, Xr, c='black', s=7)

X = np.array(list(zip(Xp,Xq,Xr)))
plt.show()

inputs=df.drop(['STATUS'],axis='columns')
inputs_n=inputs.drop(['POLICE_ZONE','WARD'],axis='columns')
print(inputs_n.head())
print(c)

X_train, X_test, y_train, y_test = train_test_split(inputs_n, t, random_state=0)

classifier = svm.SVC(kernel='linear', C=0.01).fit(X_train, y_train)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
