"""""
Jere Koski
Low-cost asset inventory level monitoring using a neural network
"""""

import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import linear_model


def calculate_regression():
    file_path = 'inventory_data'
    file = pd.read_csv(file_path, names=['datetime', 'number'],
                       parse_dates=['datetime'])

    file['year'] = file['datetime'].dt.year
    file['month'] = file['datetime'].dt.month
    file['day'] = file['datetime'].dt.day
    file['hour'] = file['datetime'].dt.hour
    file['minute'] = file['datetime'].dt.minute

    regression_model = linear_model.LinearRegression()
    x = np.array([file['year'], file['month'], file['day'],
         file['hour'], file['minute']]).transpose()

    y =  np.array(file['number'])
    reg = regression_model.fit(x, y)
    return reg

def test_prediction(nn_model, lr_model):
    scale_factor = 0.8
    dates = ['2023-12-16 06:00:00', '2023-12-16 12:00:00',
             '2023-12-16 18:00:00', '2023-10-16 06:00:00',
             '2023-10-16 12:00:00', '2023-10-16 18:00:00',
             '2023-08-16 06:00:00', '2023-08-16 12:00:00',
             '2023-08-16 18:00:00', '2035-12-16 06:00:00',
             '2035-12-16 12:00:00', '2036-12-16 18:00:00']
    for test_datetime_str in dates:
        test_datetime = datetime.strptime(test_datetime_str,
                                          '%Y-%m-%d %H:%M:%S')

        input_year = test_datetime.year
        input_month = test_datetime.month
        input_day = test_datetime.day
        input_hour = test_datetime.hour
        input_minute = test_datetime.minute

        input_features = np.array([[input_month, input_hour]])
        nn_predicted_number = nn_model.predict(input_features)[0][0]
        lr_predicted_number = lr_model.predict(np.array([[input_year, input_month,
                                      input_day, input_hour, input_minute]]))[0]


        print(f"Datetime: {test_datetime}")

        print(f"Neural network predicted number: {nn_predicted_number}")
        print(f"Linear regression predicted number: {lr_predicted_number}")

        print(f"Neural network scaled Predicted number: {nn_predicted_number * scale_factor}")
        print(f"Linear regression scaled Predicted number: {lr_predicted_number * scale_factor}")
        print()


def input_prediction(model_):
    scale_factor = 0.8

    while True:
        input_datetime_str = input("Datetime (YYYY-MM-DD HH:mm:ss): ")

        if input_datetime_str.lower() == '':
            break
        input_number = float(input("number: "))

        try:
            input_datetime = datetime.strptime(input_datetime_str,
                                               '%Y-%m-%d %H:%M:%S')
        except ValueError:
            print("Invalid datetime.")
            continue

        input_month = input_datetime.month
        input_hour = input_datetime.hour

        input_features = np.array([[input_month, input_hour]])

        predicted_number = model_.predict(input_features)[0][0]

        scaled_predicted_number = predicted_number * scale_factor

        print(f"Datetime: {input_datetime_str}")
        print(f"Predicted number: {predicted_number}")
        print(
            f"Scaled Predicted number: {scaled_predicted_number}")
        print()

        if input_number < scaled_predicted_number:
            print("The number is lower than expected.")
        else:
            print("The number is as expected or higher.")


def main():
    nn_model = tf.keras.models.load_model('inventory_model.keras')

    print()
    nn_model.summary()
    print()

    lr_model = calculate_regression()

    print()
    print(lr_model.intercept_, lr_model.coef_)
    print()

    test_prediction(nn_model, lr_model)


main()
