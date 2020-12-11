# coding=utf-8
# TODO: save parameters of the trained model
# TODO: develop test functions?

from janestreetKaggle.tools import big_csv_reader as bcr
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import os


n_points = 2000
n_input = 50
n_train = 1800
n_validation = 100
n_test = 100
file = '../jane-street-market-prediction/train.csv'
os.chdir(r'C:\Kaggle-King\janestreetKaggle')


if __name__ == '__main__':
    # as an input, we've got: 129 features, 1 date, 1 id,       1 weight,       1 + 4 responses

    # BUILD THE TRAINING SET
    n_input = 129 + 1 + 1
    sample_data = bcr.read_big_csv(filepath=file, n=100)
    y_columns = ['weight', 'resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4']
    y_data = sample_data[y_columns]
    x_data = sample_data.drop(columns=y_columns + ['ts_id'], axis=1)
    # y_hat = np.sign(y_data['resp']).astype(int)
    y_hat = y_data['resp']
    x_train = x_data.to_numpy()
    y_train = y_hat.to_numpy().reshape((100, 1))

    # BUILD THE MODEL
    model_1 = keras.models.Sequential()
    model_1.add(keras.layers.Dense(20, input_dim=n_input, activation="tanh"))
    model_1.add(keras.layers.Dense(1, activation="sigmoid"))
    model_1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mean_squared_error'])
    # model_1.build(input_shape=(131,))
    model_1.summary()
    model_1.fit(x_train, y_train, epochs=100, batch_size=10)
    _, error = model_1.evaluate(x_train, y_train)
    print('error:', error)

    x_validation, y_validation = create_input_output_set(y, n_validation, n_train + 1)
    prediction_v = model_1.predict(x_validation)

    colore = n_train * ['black'] + n_validation * ['blue'] + n_test * ['green']
    plt.scatter(x, y, s=1, c=colore)
    plt.scatter(x[n_train + 1: n_train + 1 + n_validation], prediction_v, s=1, c='red')

    x_test, y_test = create_input_output_set(y, n_test - 1, n_train + n_validation + 1)
    prediction_t = model_1.predict(x_test)
    plt.scatter(x[n_train + n_validation + 1: n_train + n_validation + 1 + n_test - 1], prediction_t, s=1, c='magenta')

    plt.show()



