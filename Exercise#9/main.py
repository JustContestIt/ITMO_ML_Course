import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


if __name__ == "__main__":
    print("Добавляем данные в файл с названием report.csv")
    n = int(input("Введите значение параметра n_clusters -> "))
    n1 = float(input("Введите первую дробь из шести -> "))
    n2 = float(input("Введите вторую дробь из шести -> "))
    n3 = float(input("Введите третью дробь из шести(не удобно, знаю. "
                     "Времени мало допиливать нормально) -> "))
    n4 = float(input("Введите четвертую дробь из шести -> "))
    n5 = float(input("Введите пятую дробь из шести -> "))
    n6 = float(input("Введите шестую дробь из шести -> "))
    mi = int(input("Введите значение параметра max_iter -> "))
    ni = int(input("Введите значение параметра n_init -> "))
    df = pd.read_csv("report.csv", delimiter=',', index_col='Object')
    df_norm = df.drop('Cluster', axis=1)
    kmeans = KMeans(n_clusters=n, init=np.array([[n1, n2], [n3, n4], [n5, n6]]), max_iter=mi, n_init=ni)
    model = kmeans.fit(df_norm)
    df_norm["Clusters"] = model.labels_.tolist()
    print("\nОтветы")
    print("------------------------------------------------------------------")
    print(df_norm)
    alldistances = kmeans.fit_transform(df_norm.drop('Clusters', axis=1))
    search = 0
    Clusters_array = []
    for i in range(len(df_norm['Clusters'])):
        if (df_norm['Clusters'][i + 1] == search):
            Clusters_array.append(alldistances[i][search])
    print("\nПоследний ответ: " + str(round(sum(Clusters_array) / len(Clusters_array), 3)))

