from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from IPython.display import Image
from imutils import paths
import numpy as np
import cv2
import os
from sklearn.metrics import f1_score


def extract_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


if __name__ == "__main__":
    imagePaths = sorted(list(paths.list_images('train')))
    data = []
    labels = []
    print("\nСкачайте архивы и разархивируйте их содержимое в папки 'train' и 'test' соответственно. "
          "Названия папок должны соответствовать этим названиям, иначе меняйте код:)")
    c = float(input("Введите значение С -> "))
    rs = int(input("Введите значение random_state -> "))
    ts = int(input("Введите второе число дроби -> "))
    x1 = int(input("Введите номер первого коэффициента -> "))
    x2 = int(input("Введите номер второго коэффициента -> "))
    x3 = int(input("Введите номер третьего коэффициента -> "))
    for (i, imagePath) in enumerate(imagePaths):
        image = cv2.imread(imagePath, 1)
        label = imagePath.split(os.path.sep)[-1].split(".")[0]
        hist = extract_histogram(image)
        data.append(hist)
        labels.append(label)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    Image(filename=imagePaths[0])
    (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data),
                                                                      labels,
                                                                      test_size=float(round(ts/100.0, 2)),
                                                                      random_state=rs)
    model = LinearSVC(random_state=rs, C=c) # попробовать значения с random_state = 9 и C=0.51
    model.fit(trainData, trainLabels)
    predictions = model.predict(testData)
    # print(classification_report(testLabels, predictions, target_names=le.classes_))

    print("\nФормат ввода названия фотографий такой -> 'dog.1028'"
          " <- то есть без '.jpg'")
    s1 = input("Введите название первой фотографии -> ")
    s2 = input("Введите название второй фотографии -> ")
    s3 = input("Введите название третьей фотографии -> ")
    s4 = input("Введите название четвертой фотографии -> ")
    singleImage = cv2.imread('test/' + s1 + '.jpg')
    histt = extract_histogram(singleImage)
    histt2 = histt.reshape(1, -1)
    prediction1 = model.predict(histt2)
    singleImage = cv2.imread('test/' + s2 + '.jpg')
    histt = extract_histogram(singleImage)
    histt2 = histt.reshape(1, -1)
    prediction2 = model.predict(histt2)
    singleImage = cv2.imread('test/' + s3 + '.jpg')
    histt = extract_histogram(singleImage)
    histt2 = histt.reshape(1, -1)
    prediction3 = model.predict(histt2)
    singleImage = cv2.imread('test/' + s4 + '.jpg')
    histt = extract_histogram(singleImage)
    histt2 = histt.reshape(1, -1)
    prediction4 = model.predict(histt2)
    print("\nОтветы")
    print("------------------------------------------------------------------")
    print("Первый ответ: " + str(round(float(model.coef_[0][x1]), 2)))
    print("Второй ответ: " + str(round(float(model.coef_[0][x2]), 2)))
    print("Третий ответ: " + str(round(float(model.coef_[0][x3]), 2)))
    predictions = model.predict(testData)
    x = f1_score(testLabels, predictions, average='macro')
    print("Четвертый ответ: " + str(round(float(x), 3)) + " (может быть не точным, "
                                                          "поэтому лучше всего будет округлять до десятых)")

    print("\nОтветы для картинок")
    print("Первая картинка: " + str(prediction1))
    print("Вторая картинка: " + str(prediction2))
    print("Третья картинка: " + str(prediction3))
    print("Четвертая картинка: " + str(prediction4))
