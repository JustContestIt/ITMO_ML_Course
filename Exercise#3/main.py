import pandas as pd
from sklearn.linear_model import LinearRegression
df = pd.read_csv("report.csv", index_col='id')
print(f'Первый ответ -> {df["X"].mean()}')
print(f'Второй ответ -> {df["Y"].mean()}')
X = pd.DataFrame(df.drop(['Y'], axis=1))
Y = pd.DataFrame(df['Y'])
reg = LinearRegression().fit(X,Y)
print(f'Третий ответ -> {reg.coef_}')
print(f'Четвертый ответ -> {reg.intercept_}')
print(f'Пятый ответ -> {reg.score(X,Y)}\n')
df2 = pd.read_csv("candy-data.csv", index_col='competitorname')
train_data = df2.drop(['Haribo Twin Snakes', 'Hersheys Krackel'])
X2 = pd.DataFrame(train_data.drop(['winpercent', 'Y'], axis=1))
Y2 = pd.DataFrame(train_data['winpercent'])
reg2 = LinearRegression().fit(X2, Y2)
candy1 = df2.loc['Haribo Twin Snakes',:].to_frame().T
print(reg2.predict(candy1.drop(['winpercent', 'Y'], axis=1)))
candy2 = df2.loc['Hersheys Krackel',:].to_frame().T
print(reg2.predict(candy2.drop(['winpercent', 'Y'], axis=1)))
print(f'Восьмой ответ -> {reg2.predict([[0, 1, 1, 1, 0, 0, 1, 0, 0, 0.32, 0.219]])}')