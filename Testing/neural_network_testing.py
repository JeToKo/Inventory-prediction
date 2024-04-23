"""""
Jere Koski
Low-cost asset inventory level monitoring using a neural network
"""""

import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import linear_model
import matplotlib.pyplot as plt

"""
Copy from neural_network.py
"""
def data():
    file_path = '../Neural network/inventory_data_v2'
    file = pd.read_csv(file_path, names=['datetime', 'number'],
                       parse_dates=['datetime'])
    file['year'] = file['datetime'].dt.year
    file['month'] = file['datetime'].dt.month
    file['day'] = file['datetime'].dt.day
    file['hour'] = file['datetime'].dt.hour
    file['minute'] = file['datetime'].dt.minute

    features = ['year', 'month', 'day', 'hour', 'minute']
    x_train = file[features]
    x_test = file[features]

    y = file['number']

    num_samples = len(file)
    train_size = int(0.8 * num_samples)

    indices = np.arange(num_samples)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_x, train_y = x_train.iloc[train_indices], y.iloc[train_indices]
    test_x, test_y = x_test.iloc[test_indices], y.iloc[test_indices]

    return train_x, test_x, train_y, test_y


def calculate_regression(train_x, train_y):
    regression_model = linear_model.LinearRegression()
    x = np.array([train_x['month'], train_x['hour']]).transpose()

    y =  np.array(train_y)
    reg = regression_model.fit(x, y)
    return reg

def test_prediction(nn_model, lr_model, test_x, test_y):
    scale_factor = 0.7

    nn_values = []
    lr_values = []
    inventory_values = []
    time = []

    current_month = 0
    current_month_values = []
    month_values = []

    nn_accuracy_values = []
    lr_accuracy_values = []

    for (year, month, day, hour, minute), inventory in zip(test_x.itertuples(index=False), test_y):
        input_features = np.array([[ month, hour]])

        nn_predicted_number = nn_model.predict(input_features)[0][0]
        lr_predicted_number = lr_model.predict(np.array([[month, hour]]))[0]


        inventory_values.append(inventory)
        nn_values.append(nn_predicted_number * scale_factor)
        lr_values.append(lr_predicted_number * scale_factor)
        time.append(datetime(year, month, day, hour, minute))

        if inventory != 0:
            accuracy = (abs(nn_predicted_number - inventory) / inventory)
            if accuracy < 0.30:
                nn_accuracy_values.append(accuracy)

        if inventory != 0:
            accuracy = (abs(lr_predicted_number - inventory) / inventory)
            if accuracy < 0.30:
                lr_accuracy_values.append(accuracy)


        if current_month != month:
            if current_month != 0:
                month_values.append(current_month_values)
            current_month = month
            current_month_values = []

        if current_month == month:
            current_month_values.append(nn_predicted_number)


        print(f"Observed inventory level: {inventory}, Month: {month}, Hour: {hour}")

        print(f"Neural network predicted number: {nn_predicted_number}")
        print(f"Linear regression predicted number: {lr_predicted_number}")

        print(f"Neural network scaled Predicted number: {nn_predicted_number * scale_factor}")
        print(f"Linear regression scaled Predicted number: {lr_predicted_number * scale_factor}")
        print()

    print(f"Total test dates: {len(inventory_values)}")
    print(f"Neural network accuracy: {(len(nn_accuracy_values) / len(inventory_values))*100}")
    print(f"Linear regression accuracy: {(len(lr_accuracy_values) / len(inventory_values))*100}")

    """
    month = 8
    for line in month_values:
        print(f'month_{month} = {line};')
        if month != 12:
            month += 1
        else:
            month = 1
    """

    plt.ylabel('Inventory level')
    plt.xlabel('Time')
    plt.plot(time, inventory_values, label='Simulated Inventory Level')
    plt.plot(time, nn_values, color='g', label='Neural Network')
    plt.legend()
    plt.show()

    plt.ylabel('Inventory level')
    plt.xlabel('Time')
    plt.plot(time, inventory_values, label='Simulated Inventory Level')
    plt.plot(time, lr_values, color='r', label='Linear Regression')
    plt.legend()
    plt.show()

    plt.ylabel('Inventory level')
    plt.xlabel('Time')
    plt.plot(time, inventory_values, label='Simulated Inventory Level')
    plt.plot(time, nn_values, color='g', label='Neural Network')
    plt.plot(time, lr_values, color='r', label='Linear Regression')


    plt.savefig('predict_graph.png')

    plt.legend()
    plt.show()


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
    nn_model = tf.keras.models.load_model(
        '../Neural network/inventory_model_v1.5.keras')

    train_x, test_x, train_y, test_y = data()

    print()
    nn_model.summary()
    print()

    lr_model = calculate_regression(train_x, train_y)

    print()
    print('Linear regression model:',lr_model.intercept_, lr_model.coef_)
    print()

    test_prediction(nn_model, lr_model, test_x, test_y)


main()