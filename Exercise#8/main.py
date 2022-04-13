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


if __name__ == "__main__":
    df = pd.read_csv('diabetes.csv')
    task_data = df.head(670)
    print(len(task_data[task_data['Outcome'] == 0]))
    train = task_data.head(int(len(task_data) * 0.8))
    test = task_data.tail(int(len(task_data) * 0.2))
    features = list(train.columns[:8])
    x = train[features]
    y = train['Outcome']
    tree = DecisionTreeClassifier(criterion='entropy',  # критерий разделения
                                  min_samples_leaf=10,  # минимальное число объектов в листе
                                  max_leaf_nodes=10,  # максимальное число листьев
                                  random_state=2020)
    clf = tree.fit(x, y)
    print(clf.tree_.max_depth)
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
    print(accuracy_score(y_true, y_pred))
    print(f1_score(y_true, y_pred, average='macro'))
    df.loc[708, features]
    print(clf.predict([df.loc[712, features].tolist()])[0])
    print(clf.predict([df.loc[749, features].tolist()])[0])
    print(clf.predict([df.loc[703, features].tolist()])[0])
    print(clf.predict([df.loc[740, features].tolist()])[0])

