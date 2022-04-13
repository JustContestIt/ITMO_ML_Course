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

    for (i, imagePath) in enumerate(imagePaths):
        image = cv2.imread(imagePath, 1)
        label = imagePath.split(os.path.sep)[-1].split(".")[0]
        hist = extract_histogram(image)
        trainData.append(hist)
        labels.append(label)

    Y = [1 if x == 'cat' else 0 for x in labels]

    # ------------------------------------------------------------------------
    tree = DecisionTreeClassifier(criterion='entropy',  # критерий разделения
                                  min_samples_leaf=10,  # минимальное число объектов в листе
                                  max_leaf_nodes=20,  # максимальное число листьев
                                  random_state=462)
    bagging = BaggingClassifier(tree,  # базовый алгоритм
                                n_estimators=19,  # количество деревьев
                                random_state=462)
    bagging.fit(trainData, Y)
    # ------------------------------------------------------------------------
    svm = LinearSVC(random_state=462, C=1.09)
    svm.fit(trainData, Y)
    # ------------------------------------------------------------------------
    forest = RandomForestClassifier(n_estimators=19,  # количество деревьев
                                    criterion='entropy',  # критерий разделения
                                    min_samples_leaf=10,  # минимальное число объектов в листе
                                    max_leaf_nodes=20,  # максимальное число листьев
                                    random_state=462)
    forest.fit(trainData, Y)
    # ------------------------------------------------------------------------
    lr = LogisticRegression(solver='lbfgs', random_state=462)
    # ------------------------------------------------------------------------
    base_estimators = [('SVM', svm), ('Bagging DT', bagging), ('DecisionForest', forest)]
    sclf = StackingClassifier(estimators=base_estimators, final_estimator=lr, cv=2)  # здесь CV, изменить
    sclf.fit(trainData, Y)
    # ------------------------------------------------------------------------
    singleImage = cv2.imread('test/dog.1023.jpg')  # вставлять сюда
    histt_1 = extract_histogram(singleImage)
    histt1 = histt_1.reshape(1, -1)
    prediction1 = sclf.predict(histt1)
    # ------------------------------------------------------------------------
    singleImage = cv2.imread('test/dog.1029.jpg')  # вставлять сюда
    histt_2 = extract_histogram(singleImage)
    histt2 = histt_2.reshape(1, -1)
    prediction2 = sclf.predict(histt2)
    # ------------------------------------------------------------------------
    singleImage = cv2.imread('test/dog.1006.jpg')  # вставлять сюда
    histt_3 = extract_histogram(singleImage)
    histt3 = histt_3.reshape(1, -1)
    prediction3 = sclf.predict(histt3)
    # ------------------------------------------------------------------------
    singleImage = cv2.imread('test/dog.1004.jpg')  # вставлять сюда
    histt_4 = extract_histogram(singleImage)
    histt4 = histt_4.reshape(1, -1)
    prediction4 = sclf.predict(histt4)
    # ------------------------------------------------------------------------
    print(sclf.score(trainData, Y))
    print(prediction1)
    print(sclf.predict_proba(histt1))
    print(prediction2)
    print(sclf.predict_proba(histt2))
    print(prediction3)
    print(sclf.predict_proba(histt3))
    print(prediction4)
    print(sclf.predict_proba(histt4))
