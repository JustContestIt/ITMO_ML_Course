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

    for (i, imagePath) in enumerate(imagePaths):
        image = cv2.imread(imagePath, 1)
        label = imagePath.split(os.path.sep)[-1].split(".")[0]
        hist = extract_histogram(image)
        data.append(hist)
        labels.append(label)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    print(labels[0])
    Image(filename=imagePaths[0])
    (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data), labels, test_size=0.25,
                                                                      random_state=7) # если что поменять
# random_state на 9
    model = LinearSVC(random_state=7, C=0.58) # попробовать значения с random_state = 9 и C=0.51
    model.fit(trainData, trainLabels)
    predictions = model.predict(testData)
    print(classification_report(testLabels, predictions, target_names=le.classes_))
    print(model.coef_[0][308])
    print(model.coef_[0][13])
    print(model.coef_[0][414])
    predictions = model.predict(testData)
    print(f1_score(testLabels, predictions, average='macro'))
    singleImage = cv2.imread('test/cat.1041.jpg')  # вставлять конкретно сюда
    histt = extract_histogram(singleImage)
    histt2 = histt.reshape(1, -1)
    prediction1 = model.predict(histt2)
    singleImage = cv2.imread('test/cat.1028.jpg')  # вставлять конкретно сюда
    histt = extract_histogram(singleImage)
    histt2 = histt.reshape(1, -1)
    prediction2 = model.predict(histt2)
    singleImage = cv2.imread('test/cat.1049.jpg')  # вставлять конкретно сюда
    histt = extract_histogram(singleImage)
    histt2 = histt.reshape(1, -1)
    prediction3 = model.predict(histt2)
    singleImage = cv2.imread('test/cat.1037.jpg')  # вставлять конкретно сюда
    histt = extract_histogram(singleImage)
    histt2 = histt.reshape(1, -1)
    prediction4 = model.predict(histt2)
    print("Вывод для картинок")
    print(prediction1)
    print(prediction2)
    print(prediction3)
    print(prediction4)
