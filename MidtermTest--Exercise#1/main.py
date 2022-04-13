# ЗАРАНЕЕ СКАЧАТЬ OPENPYXL
# это решение на 4.8/6 баллов
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv("data-1649279303550.csv")
print(df["mip"].mean())
a = df.drop(columns=["target"])
scaler = MinMaxScaler()
b = a.columns
c = scaler.fit_transform(a)
d = pd.DataFrame(c, columns=b)
print(d["mip"].mean())
writer = pd.ExcelWriter('output.xlsx')
d.to_excel(writer)
writer.close()