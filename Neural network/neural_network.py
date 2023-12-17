"""""
Jere Koski
Low-cost asset inventory level monitoring using a neural network
"""""

import tensorflow as tf
import pandas as pd
import numpy as np


def data():
    file_path = 'history_inventory_data'
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

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_x, train_y = x.iloc[train_indices], y.iloc[train_indices]
    test_x, test_y = x.iloc[test_indices], y.iloc[test_indices]

    return train_x, test_x, train_y, test_y


def neural_network(train_x, test_x, train_y, test_y, epochs, batch_size):
    input_shape = train_x.shape[1]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
              validation_data=(test_x, test_y))

    model.save('history_inventory_model.keras')

    return model


def main():
    epochs = 100000
    batch_size = 10000
    train_x, test_x, train_y, test_y = data()

    model = neural_network(train_x, test_x, train_y, test_y,
                           epochs, batch_size)

    print('***********************************')
    print(train_x)
    print('***********************************')
    print(train_y)
    print('***********************************')
    print(test_x)
    print('***********************************')
    print(test_y)
    print('***********************************')

    accuracy = model.evaluate(test_x, test_y)
    print("Accuracy: {}".format(accuracy))
    print()

    model.summary()


main()
