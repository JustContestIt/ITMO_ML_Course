import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
print("Задание 2.1")
print("Введите название файла без расширения, например '7_25'")
print("Если неправильно ввели, то программа не будет работать)")
s = input()
X = pd.read_csv(s + ".csv", header = None)
pca = PCA(n_components=2, svd_solver='full')
X_transformed = pca.fit(X).transform(X)
print("Первый ответ: " + str(round(float(X_transformed[0][0]), 3)))
print("Второй ответ: " + str(round(float(X_transformed[0][1]), 3)))
# print(X_transformed[0])
explained_variance = np.round(np.cumsum(pca.explained_variance_ratio_),3)
x = list(explained_variance)
if len(x) == 2:
    print("Третий ответ: " + str(x[1]))
else:
    print("Третий ответ: " + str(x[0]))

print("Введите верхнюю границу доли объясненной дисперсии")
s1 = float(input())
pca = PCA(n_components=10, svd_solver='auto')
X_full = pca.fit(X).transform(X)
explained_variance1 = np.round(np.cumsum(pca.explained_variance_ratio_),3)
x = list(explained_variance1)
ans = 0
for i in x:
    if float(i) < s1:
        ans += 1
    else:
        ans += 1
        break
print("Четвертый ответ: " + str(ans))
plt.plot(X_transformed[:101, 0], X_transformed[:101, 1], 'o', markerfacecolor='red', markeredgecolor='k', markersize=8)
print("Пятый ответ: это просто кол-во групп на которые разделились точки на рисунке" + "\n")
plt.show()
print("Задание 2.2")
print("Введите цифру из названия файла, например '441' из файла 'X_reduced_441.csv'")
s1 = input()
scores = np.genfromtxt('X_reduced_' + s1 + '.csv', delimiter=';')
loadings = np.genfromtxt('X_loadings_' + s1 + '.csv', delimiter=';')
values = np.dot(scores,loadings.T)
plt.imshow(values, cmap='Greys_r')
print("Смотрите рисунок")
plt.show()