import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


if __name__ == "__main__":
    df = pd.read_csv("kmeans.csv", delimiter=',', index_col='Object')
    df_norm = df.drop('Cluster', axis=1)
    kmeans = KMeans(n_clusters=3, init=np.array([[8.67, 12.67], [15.5, 9.75], [13.4, 12.6]]), max_iter=100, n_init=1)
    model = kmeans.fit(df_norm)
    df_norm["Clusters"] = model.labels_.tolist()
    print(df_norm)
    alldistances = kmeans.fit_transform(df_norm.drop('Clusters', axis=1))
    search = 0
    Clusters_array = []
    for i in range(len(df_norm['Clusters'])):
        if (df_norm['Clusters'][i + 1] == search):
            Clusters_array.append(alldistances[i][search])
    print(sum(Clusters_array) / len(Clusters_array))

