import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
X = pd.read_csv("7_25.csv", header = None)
pca = PCA(n_components=2, svd_solver='full')
X_transformed = pca.fit(X).transform(X)
print(X_transformed[0])
explained_variance = np.round(np.cumsum(pca.explained_variance_ratio_),3)
print(explained_variance)
pca = PCA(n_components=10, svd_solver='auto')
X_full = pca.fit(X).transform(X)
explained_variance1 = np.round(np.cumsum(pca.explained_variance_ratio_),3)
print(explained_variance1)
plt.plot(X_transformed[:101, 0], X_transformed[:101, 1], 'o', markerfacecolor='red', markeredgecolor='k', markersize=8)
plt.show()
scores = np.genfromtxt('X_reduced_441.csv', delimiter=';')
loadings = np.genfromtxt('X_loadings_441.csv', delimiter=';')
values = np.dot(scores,loadings.T)
plt.imshow(values, cmap='Greys_r')
plt.show()