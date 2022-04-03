import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
train_data = pd.read_csv("report.csv", index_col="id")
X = pd.DataFrame(train_data.drop(['Class'], axis=1))
y = pd.DataFrame(train_data['Class']).values.ravel()
Euclid = KNeighborsClassifier(n_neighbors=3, p=2)
Euclid.fit(X, y)
Object = [92, 85]
print(Euclid.kneighbors([Object]))
print(Euclid.predict([Object]))
Manhattan = KNeighborsClassifier(n_neighbors=3, p=1)
Manhattan.fit(X, y)
print(Manhattan.kneighbors([Object]))
print(Manhattan.predict([Object]))
