import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
df = pd.read_csv("candy-data.csv", index_col='competitorname')
train_data = df.drop(['Mike & Ike', 'Root Beer Barrels', 'Starburst'])
X = pd.DataFrame(train_data.drop(['winpercent', 'Y'], axis=1))
Y = pd.DataFrame(train_data['Y'])
reg = LogisticRegression(random_state=2019, solver='lbfgs').fit(X, Y.values.ravel())
test_data = pd.read_csv("candy-test.csv", delimiter=',', index_col='competitorname')
X_test = pd.DataFrame(test_data.drop(['Y'], axis=1))
Y_pred = reg.predict(X_test)
Y_pred_probs = reg.predict_proba(X_test)
Probability = pd.DataFrame({
    '0': [x[0] for x in Y_pred_probs],
    '1': [x[1] for x in Y_pred_probs]
},
index=test_data.index)
print(Probability)
Y_true = (test_data['Y'].to_frame().T).values.ravel()
fpr, tpr, _ = metrics.roc_curve(Y_true, Y_pred)
Y_pred_probs_class_1 = Y_pred_probs[:, 1]
print(metrics.recall_score(Y_true, Y_pred))
print(metrics.precision_score(Y_true, Y_pred))
print(metrics.roc_auc_score(Y_true, Y_pred_probs_class_1))
