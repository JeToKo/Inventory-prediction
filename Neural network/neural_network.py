"""""
Jere Koski
Low-cost asset inventory level monitoring using a neural network
"""""
import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime


def data():
    file_path = 'inventory_data'
    file = pd.read_csv(file_path, names=['datetime', 'number'],
                       parse_dates=['datetime'])

    file['month'] = file['datetime'].dt.month
    file['hour'] = file['datetime'].dt.hour

    features = ['month', 'hour']
    x = file[features]
    y = file['number']

    num_samples = len(file)
    train_size = int(0.8 * num_samples)

    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_x, train_y = x.iloc[train_indices], y.iloc[train_indices]
    test_x, test_y = x.iloc[test_indices], y.iloc[test_indices]

    return train_x, test_x, train_y, test_y


def neural_network(train_x, test_x, train_y, test_y, epochs, batch_size):
    input_shape = train_x.shape[1]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation='relu'),
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
              validation_data=(test_x, test_y))

    model.save('inventory_model.keras')

    return model


def test_prediction(model_):
    scale_factor = 0.8
    dates = ['2023-12-16 06:00:00', '2023-12-16 12:00:00',
             '2023-12-16 18:00:00', '2023-10-16 06:00:00',
             '2023-10-16 12:00:00', '2023-10-16 18:00:00',
             '2023-08-16 06:00:00', '2023-08-16 18:00:00',
             '2023-08-16 18:00:00', '2035-12-16 06:00:00',
             '2035-12-16 12:00:00', '2036-12-16 18:00:00']
    for test_datetime_str in dates:
        test_datetime = datetime.strptime(test_datetime_str,
                                           '%Y-%m-%d %H:%M:%S')

        input_month = test_datetime.month
        input_hour = test_datetime.hour

        input_features = np.array([[input_month, input_hour]])

        predicted_number = model_.predict(input_features)[0][0]

        scaled_predicted_number = predicted_number * scale_factor

        print(f"Datetime: {test_datetime}")
        print(f"Predicted Number: {predicted_number}")
        print(f"Scaled Predicted Number: {scaled_predicted_number}")
        print()


def input_prediction(model_):
    scale_factor = 0.8

    while True:
        input_datetime_str = input("Datetime (YYYY-MM-DD HH:mm:ss): ")

        if input_datetime_str.lower() == '':
            break
        input_number = float(input("Number: "))

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
        print(f"Predicted Number: {predicted_number}")
        print(
            f"Scaled Predicted Number: {scaled_predicted_number}")
        print()

        if input_number < scaled_predicted_number:
            print("The number is lower than expected.")
        else:
            print("The number is as expected or higher.")


def main():
    epochs = 200000
    batch_size = 1000
    train_x, test_x, train_y, test_y = data()

    model = neural_network(train_x, test_x, train_y, test_y,
                           epochs, batch_size)

    accuracy = model.evaluate(test_x, test_y)

    print("Accuracy: {}".format(accuracy))
    print()

    model.summary()
    print()

    test_prediction(model)


main()
