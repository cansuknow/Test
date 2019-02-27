
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

def data_preprocessing(file_path):
    data = pd.read_csv(file_path)
    data.dropna()
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

    df.drop(['CustomerID', 'SalesMail'], axis=1, inplace=True)
    df.drop_duplicates(inplace=True)
    return df


def regression_model(data_frame):
    target = data_frame['SalesPrice']
    inputs = data_frame.drop('SalesPrice', axis=1)
    X = inputs
    Y = target
    model = lm.LinearRegression()
    result = model.fit(X, Y)
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    predictions = model.predict(X)
    y_true = target.fillna("ffill")
    y_pred = predictions
    return model.summary(), mean_absolute_error(y_true, y_pred), mean_squared_error(y_true, y_pred)

print(regression_model(data_preprocessing("/Users/cansuyuksel/Desktop/test_t.csv")))



birthday = input("Doğum yılınız: ")
gender = input("Cinsiyetiniz: ")
city = input("Doğum yeriniz: ")
degree_level = input("Eğitim durumunuz: ")
school_name = input("Mezun olduğunuz okul: ")
sales_date = input("Satın alma yılı: ")
payment_type = input("Ödeme şekliniz: ")
product_type = input("Talep edilen ürünün adı: ")
base_name = input("Bağlı olduğu merkez: ")

newData = pd.DataFrame([{"Birthday": birthday,
                         "Gender": gender,
                         "City": city ,
                         "DegreeLevel": degree_level,
                         "SchoolName": school_name,
                         "SalesDate": sales_date,
                         "PaymentType": payment_type,
                         "ProductName": product_type,
                         "BaseName": base_name
                         }])

print(newData)

def preprocess_input(input_df):
    df_pred = pd.get_dummies(input_df,columns=['Birthday',
                                       'Gender',
                                       'City',
                                       'DegreeLevel',
                                       'SchoolName',
                                       'SalesDate',
                                       'PaymentType',
                                       'ProductName',
                                       'BaseName'
                                       ])
    ready_pred = pd.concat([df,df_pred],ignore_index=True,sort=False).fillna(0)
    ready_pred.drop("SalesPrice", axis=1, inplace=True)
    return ready_pred


def predict_price(ready_pred):
    new_input = ready_pred.iloc[-1]
    try:
        model.predict([new_input])
    except:
        print("Yeterli veri bulunmamaktadır.")

    return lm.predict([new_input])


print(predict_price(preprocess_input(newData)))
