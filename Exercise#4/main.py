import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
print("Возьмите данные из задания и закиньте их в файл 'report.csv'")
print("Введите первую координату:")
a = int(input())
print("Введите вторую координату:")
b = int(input())
print("Введите значение k:")
k = int(input())
train_data = pd.read_csv("report.csv", index_col="id")
X = pd.DataFrame(train_data.drop(['Class'], axis=1))
y = pd.DataFrame(train_data['Class']).values.ravel()
Euclid = KNeighborsClassifier(n_neighbors=k, p=2)
Euclid.fit(X, y)
Object = [a, b]
x = Euclid.kneighbors([Object])[0].tolist()
print("Первый ответ: " + str(round(float(x[0][0]), 3)))
x1 = Euclid.kneighbors([Object])[1].tolist()
x2 = []
for i in x1[0]:
    x2.append(int(i) + 1)
print("Второй ответ: " + str(x2))
print("Третий ответ: " + str(int(Euclid.predict([Object]))))
Manhattan = KNeighborsClassifier(n_neighbors=k, p=1)
Manhattan.fit(X, y)
x4 = Manhattan.kneighbors([Object])[0].tolist()
print("Четвертый ответ: " + str(round(float(x4[0][0]), 3)))
x5 = Manhattan.kneighbors([Object])[1].tolist()
x6 = []
for i in x5[0]:
    x6.append(int(i) + 1)
print("Пятый ответ: " + str(x6))
print("Шестой ответ: " + str(int(Manhattan.predict([Object]))) + "\n")
print("Игнорируйте ошибки")
