from IPython.display import Image
from imutils import paths
import numpy as np
import cv2
import os
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier


def extract_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


if __name__ == "__main__":
    imagePaths = sorted(list(paths.list_images('train')))
    trainData = []
    labels = []
    c = 0.0
    rs = 0
    s = ''
    s1 = ''
    criterion = ''
    msl = 0
    mln = 0
    n = 0
    cv = 0
    solver = ''

    print("\nСкачайте архивы и разархивируйте их содержимое в папки 'train' и 'test' соответственно. "
          "Названия папок должны соответствовать этим названиям, иначе меняйте код:)")
    c = float(input("Введите значение С -> "))
    s = input("Введите параметры для бэггинга в таком виде 'criterion min_samples_leaf max_leaf_nodes "
              "random_state n_estimators'\n Пример: entropy 10 20 462 19 -> ")
    solver = input("Введите значение solver -> ")
    cv = int(input("Введите значение CV -> "))
    s1 = input("Введите список названий картинок в таком виде 'dog.1001 dog.1002 cat.1003 cat.1004' -> ")
    print("\nПодожди немного\n")
    s1 = s1.split()
    s = s.split()

    for (i, imagePath) in enumerate(imagePaths):
        image = cv2.imread(imagePath, 1)
        label = imagePath.split(os.path.sep)[-1].split(".")[0]
        hist = extract_histogram(image)
        trainData.append(hist)
        labels.append(label)

    Y = [1 if x == 'cat' else 0 for x in labels]

    # ------------------------------------------------------------------------
    tree = DecisionTreeClassifier(criterion=s[0],  # критерий разделения
                                  min_samples_leaf=int(s[1]),  # минимальное число объектов в листе
                                  max_leaf_nodes=int(s[2]),  # максимальное число листьев
                                  random_state=int(s[3]))
    bagging = BaggingClassifier(tree,  # базовый алгоритм
                                n_estimators=int(s[4]),  # количество деревьев
                                random_state=int(s[3]))
    bagging.fit(trainData, Y)
    # ------------------------------------------------------------------------
    svm = LinearSVC(random_state=rs, C=c)
    svm.fit(trainData, Y)
    # ------------------------------------------------------------------------
    forest = RandomForestClassifier(n_estimators=int(s[4]),  # количество деревьев
                                    criterion=s[0],  # критерий разделения
                                    min_samples_leaf=int(s[1]),  # минимальное число объектов в листе
                                    max_leaf_nodes=int(s[2]),  # максимальное число листьев
                                    random_state=int(s[3]))
    forest.fit(trainData, Y)
    # ------------------------------------------------------------------------
    lr = LogisticRegression(solver=solver, random_state=int(s[3]))
    # ------------------------------------------------------------------------
    base_estimators = [('SVM', svm), ('Bagging DT', bagging), ('DecisionForest', forest)]
    sclf = StackingClassifier(estimators=base_estimators, final_estimator=lr, cv=cv)  # здесь CV, изменить
    sclf.fit(trainData, Y)
    # ------------------------------------------------------------------------
    singleImage = cv2.imread('test/' + s1[0] + '.jpg')  # вставлять сюда
    histt_1 = extract_histogram(singleImage)
    histt1 = histt_1.reshape(1, -1)
    prediction1 = sclf.predict(histt1)
    # ------------------------------------------------------------------------
    singleImage = cv2.imread('test/' + s1[1] + '.jpg')  # вставлять сюда
    histt_2 = extract_histogram(singleImage)
    histt2 = histt_2.reshape(1, -1)
    prediction2 = sclf.predict(histt2)
    # ------------------------------------------------------------------------
    singleImage = cv2.imread('test/' + s1[2] + '.jpg')  # вставлять сюда
    histt_3 = extract_histogram(singleImage)
    histt3 = histt_3.reshape(1, -1)
    prediction3 = sclf.predict(histt3)
    # ------------------------------------------------------------------------
    singleImage = cv2.imread('test/' + s1[3] + '.jpg')  # вставлять сюда
    histt_4 = extract_histogram(singleImage)
    histt4 = histt_4.reshape(1, -1)
    prediction4 = sclf.predict(histt4)
    # ------------------------------------------------------------------------

    print("Ответы\n------------------------------------------------------------------------")
    print("Первое задание: " + str(sclf.score(trainData, Y)) + "\n")
    print("Второе задание:")
    print("Первая картинка: " + str(round(float(sclf.predict_proba(histt1)[0][1]), 3)))
    print("Вторая картинка: " + str(round(float(sclf.predict_proba(histt2)[0][1]), 3)))
    print("Третья картинка: " + str(round(float(sclf.predict_proba(histt3)[0][1]), 3)))
    print("Четвертая картинка: " + str(round(float(sclf.predict_proba(histt4)[0][1]), 3)))
