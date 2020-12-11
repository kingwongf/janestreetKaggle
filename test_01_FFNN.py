# coding=utf-8
# TODO: save parameters of the trained model
# TODO: develop test functions?

from janestreetKaggle.tools import big_csv_reader as bcr
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd

n = 200
n_train = 190
n_dev = n - n_train
file = '../jane-street-market-prediction/train.csv'
os.chdir(r'C:\Kaggle-King\janestreetKaggle')

n_points = 2000
n_input = 50
n_train = 1800
n_validation = 100
n_test = 100

if __name__ == '__main__':
    # as an input, we've got: 129 features, 1 date, 1 id,       1 weight,       1 + 4 responses

    # PRE-PROCESS DATA
    n_input = 129 + 1 + 1
    sample_data = bcr.read_big_csv(filepath=file, n=n)
    nan_col = [0, 7, 8, 11, 12, 17, 18, 21, 22, 27, 28, 31, 31, 55, 72, 74, 78, 80, 84, 86, 90, 92, 96, 98, 102,
               104, 108, 110, 114, 116, 120, 121]
    nan_columns = ['date'] + ['feature_'+str(x) for x in nan_col]
    y_columns = ['weight', 'resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4']
    y_data = sample_data[y_columns]
    x_data = sample_data.drop(columns=y_columns + nan_columns + ['ts_id'], axis=1)
    # y_hat = np.sign(y_data['resp']).astype(int)
    x_data = np.nan_to_num(x_data.to_numpy())
    y_hat = y_data['resp'].to_numpy().reshape(n, 1)

    # TRAIN SET
    x_train, y_train = x_data[0:n_train], y_hat[0:n_train]
    # DEV SET
    x_dev, y_dev = x_data[n_train:n_train + n_dev], y_hat[n_train:n_train + n_dev]

    # BUILD THE MODEL
    model_1 = keras.models.Sequential()
    model_1.add(keras.layers.Dense(20, input_dim=x_train.shape[1], activation="tanh"))
    model_1.add(keras.layers.Dense(1, activation="tanh"))
    model_1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_crossentropy'])
    # model_1.build(input_shape=(131,))
    model_1.summary()
    model_1.fit(x_train, y_train, epochs=10, batch_size=2)
    _, error = model_1.evaluate(x_train, y_train)
    print('error:', error)

    # TEST ON THE DEV SET
    prediction = model_1.predict(x_dev)
    data = np.concatenate([prediction, y_dev], axis=1)
    comparison = pd.DataFrame(data=data, columns=['prediction', 'y_hat'])

    data_2 = np.concatenate([np.sign(prediction).astype(int), np.sign(y_dev).astype(int)], axis=1)
    comparison_2 = pd.DataFrame(data=data_2, columns=['prediction', 'y_hat'])
