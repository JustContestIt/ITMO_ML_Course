import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    df = pd.read_csv('diabetes.csv')
    xx = int(input("Введите кол-во первых строк нужных данных -> "))
    task_data = df.head(xx)
    x2 = int(input("Введите номер класса в первом задании (0 или 1) -> "))
    x1 = int(input("Введите первое число в отношении тренировочного набора данных к тестовым, например 80/20 "
                   "- введите число 80 -> "))
    train = task_data.head(int(len(task_data) * round(float(x1 / 100.0), 3)))
    test = task_data.tail(int(len(task_data) * round(float((100.0 - x1)/100.0), 3)))
    features = list(train.columns[:8])
    x = train[features]
    y = train['Outcome']
    s = input("Введите значение параметра criterion без кавычек -> ")
    mln = int(input("Введите значение параметра max_leaf_nodes -> "))
    msl = int(input("Введите значение параметра min_samples_leaf -> "))
    rs = int(input("Введите значение параметра random_state -> "))
    n1 = int(input("Введите номер первого пациента -> "))
    n2 = int(input("Введите номер второго пациента -> "))
    n3 = int(input("Введите номер третьего пациента -> "))
    n4 = int(input("Введите номер четвертого пациента -> "))
    tree = DecisionTreeClassifier(criterion=s,  # критерий разделения
                                  min_samples_leaf=msl,  # минимальное число объектов в листе
                                  max_leaf_nodes=mln,  # максимальное число листьев
                                  random_state=rs)
    clf = tree.fit(x, y)
    columns = list(x.columns)
    export_graphviz(clf, out_file='tree.dot',
                    feature_names=columns,
                    class_names=['0', '1'],
                    rounded=True, proportion=False,
                    precision=2, filled=True, label='all')

    with open('tree.dot') as f:
        dot_graph = f.read()
    graphviz.Source(dot_graph)
    features = list(test.columns[:8])
    x = test[features]
    y_true = test['Outcome']
    y_pred = clf.predict(x)
    print("\nОтветы")
    print("------------------------------------------------------------------")
    print("Первый ответ: " + str(len(task_data[task_data['Outcome'] == x2])))
    print("Второй ответ: " + str(clf.tree_.max_depth) + "\n")
    print("Для картинки выбрать снизу текст (BMI, Pregnancies и тд) и у него же значение после <="
          "\nЭто будут третий и четвертый ответы\n")
    print("Пятый ответ: " + str(round(float(accuracy_score(y_true, y_pred)), 3)))
    print("Шестой ответ: " + str(f1_score(y_true, y_pred, average='macro')))
    print("Для числа " + str(n1) + " ответ: " + str(clf.predict([df.loc[n1, features].tolist()])[0]))
    print("Для числа " + str(n2) + " ответ: " + str(clf.predict([df.loc[n2, features].tolist()])[0]))
    print("Для числа " + str(n3) + " ответ: " + str(clf.predict([df.loc[n3, features].tolist()])[0]))
    print("Для числа " + str(n4) + " ответ: " + str(clf.predict([df.loc[n4, features].tolist()])[0]))

