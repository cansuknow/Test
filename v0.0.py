import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


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

lm = linear_model.LinearRegression()
model = lm.fit(X, Y)

predictions = lm.predict(X)

y_true = target.fillna("ffill")
y_pred = predictions

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


def preprocess_input(input_df):
    df_pred = pd.get_dummies(input_df, columns=['Birthday',
                                                'Gender',
                                                'City',
                                                'DegreeLevel',
                                                'SchoolName',
                                                'SalesDate',
                                                'PaymentType',
                                                'ProductName',
                                                'BaseName'
                                                ])
    ready_pred = pd.concat([df, df_pred], ignore_index=True, sort=False).fillna(0)
    ready_pred.drop("SalesPrice", axis=1, inplace=True)
    return ready_pred


def predict_price(ready_pred):
    new_input = ready_pred.iloc[-1]
    try:
        lm.predict([new_input])
    except:
        print("Yeterli veri bulunmamaktadır.")

    return lm.predict([new_input])


print(predict_price(preprocess_input(newData)))





