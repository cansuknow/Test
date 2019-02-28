import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

data = pd.read_csv("/Users/cansuyuksel/Desktop/test_t.csv")

data = data.dropna()

data = pd.DataFrame({
    "CustomerID": data["CustomerID"],
    "Birthday": pd.DatetimeIndex(data['Birthday']).year,
    "Gender": data.Gender.map(lambda x: 'Kadın' if x == True else 'Erkek'),
    "City": data["CityName"],
    "DegreeLevel": data["DegreeLevel"],
    "SchoolName": data["SchoolName"],
    "SalesDate": pd.DatetimeIndex(data['SalesDate']).year,
    "PaymentType": data["PaymentType"],
    "ProductName": data["ProductName"],
    "SalesPrice": data["SalesPrice"],
    "BaseName": data["BaseName"],
    "SalesMail": data["SalesMail"]
})

df = pd.get_dummies(data, columns=['Birthday',
                                   'Gender',
                                   'City',
                                   'DegreeLevel',
                                   'SchoolName',
                                   'PaymentType',
                                   'ProductName',
                                   'BaseName',
                                   'SalesDate'])

print(df)

df.drop(['CustomerID', 'SalesMail'], axis=1, inplace=True)

df.drop_duplicates(inplace=True)
print(df)

target = df['SalesPrice']
inputs = df.drop('SalesPrice', axis=1)

X = inputs

Y = target

model = linear_model.LinearRegression()
model.fit(X, Y)

predictions = model.predict(X)
y_true = target
y_pred = predictions

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

print(mean_absolute_error(y_true, y_pred))
print(mean_squared_error(y_true, y_pred))

newData = pd.DataFrame([{"Birthday": "1985",
                         "Gender": "Erkek",
                         "City": "ANKARA",
                         "DegreeLevel": "Yüksek Lisans",
                         "SchoolName": "Istanbul Teknik Üniversitesi",
                         "SalesDate": "2017",
                         "PaymentType": "HAVALE",
                         "ProductName": "Proje Yönetimi ve PMP Sınavına Hazırlık",
                         "BaseName": "Ankara"
                         }])

print(newData)

df2 = pd.get_dummies(newData, columns=['Birthday',
                                       'Gender',
                                       'City',
                                       'DegreeLevel',
                                       'SchoolName',
                                       'SalesDate',
                                       'PaymentType',
                                       'ProductName',
                                       'BaseName'
                                       ])
df3 = df2.copy()
newX = list(inputs.columns)
for item in newX:
    df3[item] = 0
df3 = df3[inputs.columns]
pred_df = pd.concat([df3,df2],ignore_index=True, sort=False).fillna(0)
print(pred_df["Birthday_1985"].head())
input_X = list(pred_df.iloc[1])
print(model.predict([input_X]))

