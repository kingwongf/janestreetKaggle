# coding=utf-8
# TODO: save parameters of the trained model
# TODO: develop test functions?
# TODO: include the feature matrix somehow in the data preprocessing (computing effectively the mean of the feature can be a strategy)
# TODO: include resp1, ... , resp4

# CONCLUSION: USING THE TANH NETWORK CAN GIVE YOU SCORES THAT ARE 0, BECASUE THE SCORE FUNCTION CAN GO NEGATIVE EVEN AFTER 300.000 ITERATIONS.
# EVEN IF THE MODEL HAS LOW BIAS AND LOW VARIANCE, WHEN THE METRIC IS REPLACED WITH THE EVALUATION SCORE, THIS GOES TO 0

from janestreetKaggle.tools import big_csv_reader as bcr
from tensorflow import keras
import numpy as np
import os
from matplotlib import pyplot as plt

n = 300000
n_train = 290000
n_dev = n - n_train
epochs = 1
batch_size = 1
loss = 'MeanSquaredError'
metrics = ['MeanAbsoluteError']

working_dir = 'C:/Kaggle-King'
os.chdir(working_dir)
models_dir = working_dir + '/janestreetKaggle/nn/saved_models/'
file = working_dir + '/jane-street-market-prediction/train.csv'


def compute_scores(edge_0: int, edge_1: int, actions):
    results = y_data[['resp', 'weight']].iloc[edge_0:edge_1]
    results['action'] = actions
    current_score = score(r=results)
    results['action'] = (results['resp'] > 0)*1
    max_possible_score = score(r=results)
    print('max score:', max_possible_score, ' -- this score:', current_score, ' -- %:', current_score/max_possible_score*100)
    return current_score, max_possible_score


def score(r):
    s = sum(r.prod(axis=1))  # to be refined by date
    p = np.array([s])
    t = p.sum() / np.sqrt((p * p).sum()) * np.sqrt(250 / len(p))
    u = min(max(0, t), 6) * p.sum()
    return u


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
    hidden_neurons = [64, 32, 16]
    input_m = keras.layers.Input(shape=(x_train.shape[1],))
    m = keras.layers.BatchNormalization()(input_m)
    m = keras.layers.Dropout(rate=0.15, seed=3)(m, training=True)
    for hl in hidden_neurons:
        m = keras.layers.Dense(units=hl)(m)
        m = keras.layers.BatchNormalization()(m, training=True)
        m = keras.layers.Activation(activation='tanh')(m)
        m = keras.layers.Dropout(rate=0.15, seed=3)(m, training=True)
    m = keras.layers.Dense(1)(m)
    output_m = keras.layers.Activation(activation='tanh')(m)

    model_2 = keras.models.Model(inputs=input_m, outputs=output_m)
    model_2.compile(loss=loss, optimizer='Adam', metrics=metrics)
    model_2.summary()
    model_2.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    model_2.save(models_dir + 'model_01_with_tanh')
    _, error = model_2.evaluate(x_train, y_train)
    print('bias:', error)
    dev_loss, dev_acc = model_2.evaluate(x_dev, y_dev)
    print("Test accuracy", dev_acc)
    print("Test loss", dev_loss)
    # TEST VARIANCE ON THE DEV SET
    prediction = model_2.predict(x_dev)
    # now it compares the output after converting to binary classes
    binary_prediction = (prediction > 0)*1
    binary_y_dev = (y_dev > 0)*1
    outcome = (binary_prediction - binary_y_dev)*1

    # compute the scores on the train set and then on the dev set (weight, resp, action):
    prediction_ot = model_2.predict(x_train)
    # now it compares the output after converting to binary classes
    binary_prediction_ot = (prediction_ot > 0)*1
    binary_y_train = (y_train > 0)*1

    print('score on training set')
    compute_scores(edge_0=0, edge_1=n_train, actions=binary_prediction_ot)
    print('score on dev set')
    compute_scores(edge_0=n_train, edge_1=n_train + n_dev, actions=binary_prediction)

    plt.figure()
    plt.plot(prediction, label='prediction'), plt.plot(y_dev, label='y_dev')
    plt.legend(), plt.show()

    plt.figure()
    plt.plot(outcome, 'ro', ms=1, label='prediction - truth')
    plt.legend(), plt.show()


# todo: normalise and use sigmoid. We are obtaining always negative outputs...
# todo: QUESTION: to compute the final score, we need the 'resp', but they are not going to provide it to us?...
# todo: find the optimal threshold to tune the network
# todo: find a way to train hyperparam efficiently
# todo: compute the final score from Kaggle
# todo: dropout is ok to use?
# todo: hyperparam tuning (from lessons).
# todo: review code from King
# todo: RESTnet
# todo: look at King's version 4/4
# todo: find out why Hyperparam from King crashes with memory
# todo: change BatchNormalization
# todo: change loss function
# todo: raise the number of neurons
# todo: build network with connections (internal)
# todo: use reinforced learning
# todo: link the feature 0 (only -1 / 1) to the last layer in a way we compare only integers (use sigmoids...?)
# todo: suppress lines where weight is 0
