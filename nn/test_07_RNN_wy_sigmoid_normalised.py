# coding=utf-8

# todo: CHECK MORE CAREFULLY IF USING FIT IN INCREMENTAL STEPS IS REALLY WORKING
#  correlate the results (resp) to the very high_variance_features ?
#  how to normalize data that will come in the future? Which will be then minima and maxima?
#  worth to suppress rows with 0-weight in a RNN?
#  further simplification to be done aggregating features in tags?
#  worth to delete raw data after each batch, to save memory?
#  worth to find out how many 'nans' in the feature data?
#  introduce a dev set
#  introduce a final layer trasforming the ouputs back to "resp"
#  use a different normalization of features (with mu and sigma)
#  make a graph of the original resps?

import numpy as np
import pandas as pd
import janestreetKaggle.tools.big_csv_reader as bcr
from tensorflow import keras
import os

rnn_to_use = r'./nn/saved_models/RNN_06_test_4'
train, predict = False, True
use_test_size = True
plot_consistency_check = False

if use_test_size:
    # HOT_VECTOR_SIZE = 84
    # BATCH_SIZE = 5000
    # MINIMUM_SEQUENCE_LENGHT, MAXIMUM_SEQUENCE_LENGHT = (150, 500)
    # N_BATCH_SAMPLES = 50
    # N_LENGHTS = 10
    # N_RECORDS = 6000
    # N_TRAIN = 5000
    # N_TEST = N_RECORDS - N_TRAIN
    HOT_VECTOR_SIZE = 84
    BATCH_SIZE = 100000     # 5000
    MINIMUM_SEQUENCE_LENGHT, MAXIMUM_SEQUENCE_LENGHT = (150, 500)
    N_BATCH_SAMPLES = 2000      # 50
    N_LENGHTS = 50      # 10
    N_RECORDS = 200000  # 6000
    N_TRAIN = 190000     # 5000
    N_TEST = N_RECORDS - N_TRAIN
else:
    HOT_VECTOR_SIZE = 84
    BATCH_SIZE = 500000
    MINIMUM_SEQUENCE_LENGHT, MAXIMUM_SEQUENCE_LENGHT = (150, 500)
    N_BATCH_SAMPLES = 10000
    N_LENGHTS = 20
    N_RECORDS = 2390490
    N_TRAIN = 2370490
    N_TEST = N_RECORDS - N_TRAIN
y_columns = ['resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4']


def sigmoid_dir(x):
    # maps to the range [-1, +1]
    sigma = 2*(1/(1 + np.exp(-x)) - 0.5)
    return sigma


def sigmoid_inv(x):
    inv = - np.log(2/(x + 1) - 1)
    return inv


def build_the_model():
    model_1 = keras.Sequential()
    model_1.add(keras.layers.LSTM(128, input_shape=(None, 84)))
    model_1.add(keras.layers.Dense(5, activation='sigmoid'))
    model_1.compile(loss='MeanSquaredError', optimizer='adam', metrics=['accuracy'])
    print(model_1.summary())
    return model_1


def load_data_set(start: int = None, n: int = N_TRAIN + N_TEST):
    data = bcr.read_big_csv(filepath=bcr.file, s=start, n=n)
    c = list(data.columns)
    x_columns = sorted(list(set(c) - set(y_columns)))
    y_data = data[y_columns]
    x_data = data[x_columns]
    return x_data, y_data


def suppress_correlated_features(df: pd.DataFrame):
    f = open(r'C:\Kaggle-King\janestreetKaggle\tools\features_not_correlated.txt',
             'rt')  # load the list of correlated features
    ncf = f.read().splitlines()
    f.close()
    df = df[['weight', 'feature_0'] + ncf]  # suppress the non interesting columns
    return df


def fill_nan(df: pd.DataFrame):
    stats = pd.read_csv(r'C:\Kaggle-King\janestreetKaggle\tools\feature_stats.csv',
                        index_col=0)    # load the file with the averages
    for c in df.columns:    # choose the appropriate ones to fill the Nans
        df[c].fillna(stats.loc[c]['average'], inplace=True)
    return df


def load_and_preprocess_dataset(s, n):
    print('LOADING... ', end='')
    x, y = load_data_set(start=s, n=n)
    print('PURGING... ', end='')
    x = suppress_correlated_features(x)
    x = fill_nan(x)
    print('WEIGHTING Y HATS... ', end='')
    y = y.multiply(x['weight'], axis='index')  # weight all the resp and consider them as the effective y_hats
    print('NORMALISING NOW X AND Y_HATS')
    x = normalise_data(x, r'C:\Kaggle-King\janestreetKaggle\tools\feature_stats.csv')
    y = sigmoid_dir(y)
    return x, y


def normalise_data(df: pd.DataFrame, n_filename: str):
    # (review lessons). Also: MAYBE WORTH TO AVOID TURNING "-1" IN FEATURE_0 TO "0"
    stats = pd.read_csv(n_filename, index_col=0)        # load the file with the minima & maxima
    for c in df.columns:    # normalise each column, except for weights and resps (map between 0 and 1)
        x_min, x_max = stats.loc[c]['minima'], stats.loc[c]['maxima']
        a = 1/(x_max - x_min)
        b = -x_min*a
        df[c] = a*df[c] + b
    return df


def prepare_one_hot_vectors_input_set(df_x: pd.DataFrame, df_y: pd.DataFrame,
                                      sample_size: int, start: int):
    x_training_sample = df_x.iloc[start: start + sample_size].to_numpy()
    y_hat = df_y.iloc[start + sample_size - 1].to_numpy()
    tracking = (start, sample_size, y_hat)
    return tracking, x_training_sample, y_hat


def prepare_multiple_hot_vectors_input_set(df_x: pd.DataFrame, df_y: pd.DataFrame,
                                           start_points: np.array, sample_size: int):
    number_of_samples = len(start_points)
    x_training_set = np.zeros((number_of_samples, sample_size, df_x.shape[1]))
    y_training_set = np.zeros((number_of_samples, df_y.shape[1]))
    for p in range(number_of_samples):
        _, one_input_sequence, y_hat = prepare_one_hot_vectors_input_set(df_x=df_x, df_y=df_y,
                                                                         sample_size=sample_size, start=start_points[p])
        x_training_set[p], y_training_set[p] = one_input_sequence, y_hat
    return x_training_set, y_training_set


def prepare_full_training_set(df_x: pd.DataFrame, df_y: pd.DataFrame,
                              number_of_samples_per_lenght: int, number_of_sample_lenghts: int):
    print('ORGANISING DATA IN APPROPRIATE SETS... ')
    # WARNING: the number (number_of_sample_lenghts*number_of_samples_per_lenght) should not exceed the size of the allowed interval
    full_x_training_set, full_y_training_set = [], []
    sample_lenghts = np.random.choice(range(MINIMUM_SEQUENCE_LENGHT, MAXIMUM_SEQUENCE_LENGHT),
                                      size=number_of_sample_lenghts,
                                      replace=False)
    start_points = []
    for lenght in sample_lenghts:
        start_points_segment = np.random.choice(range(len(df_x)-lenght),
                                                size=number_of_samples_per_lenght,
                                                replace=False)
        start_points.extend(start_points_segment)
    start_points = np.asarray(start_points)
    for n_l in range(number_of_sample_lenghts):
        start_points_chunk = start_points[number_of_samples_per_lenght*n_l: number_of_samples_per_lenght*(n_l + 1)]
        x_training_chunk, y_training_chunk = prepare_multiple_hot_vectors_input_set(df_x=df_x, df_y=df_y, 
                                                                                    start_points=start_points_chunk, 
                                                                                    sample_size=sample_lenghts[n_l])
        full_x_training_set.append(x_training_chunk)
        full_y_training_set.append(y_training_chunk)
    return full_x_training_set, full_y_training_set


def prepare_full_test_set(df_x: pd.DataFrame, df_y: pd.DataFrame):
    print('ORGANISING DATA IN APPROPRIATE SETS... ')
    test_list_x, test_list_y = [], []
    test_size = len(df_x)
    for p in range(test_size - MINIMUM_SEQUENCE_LENGHT):
        sample_size = np.random.randint(MINIMUM_SEQUENCE_LENGHT, min(MAXIMUM_SEQUENCE_LENGHT, test_size - p))
        x__test, y__test = prepare_multiple_hot_vectors_input_set(df_x=df_x, df_y=df_y,
                                                                  sample_size=sample_size,
                                                                  start_points=[p])
        test_list_x.append(x__test)
        test_list_y.append(y__test)
    return test_list_x, test_list_y


def convert_back_pred_y_hat_to_prob_hat_inputlike(w_df: np.array, y_df: np.array):
    w_original_resp = sigmoid_inv(y_df)

    stats = pd.read_csv(r'C:\Kaggle-King\janestreetKaggle\tools\feature_stats.csv', index_col=0)        # load the file with the minima & maxima
    w_min, w_max = stats.loc['weight']['minima'], stats.loc['weight']['maxima']
    a = 1 / (w_max - w_min)
    b = -w_min * a
    w_original = (w_df - b)/a
    original_resp = w_original_resp / w_original

    return w_original, original_resp


def evaluate_kaggle_score(w, y_inv):
    y_inverted[np.isneginf(y_inv)] = 0
    y_inverted[np.isnan(y_inv)] = 0
    probabilities = w * y_inv * (y_inv > 0)
    probabilities_sum = probabilities.sum()
    probabilities_var = np.var(probabilities)
    t = probabilities_sum/probabilities_var * np.sqrt(250/1)
    u = min(max(t, 0), 6)*probabilities_sum
    return u


if __name__ == '__main__':

    if train:
        np.random.seed(3)          # ensures the repeatability of the training
        model = build_the_model()
        number_of_batches = N_TRAIN // BATCH_SIZE + bool(N_TRAIN % BATCH_SIZE)
        for batch in range(number_of_batches):
            print('TRAINING NOW BATCH %d / %d' % (batch + 1, number_of_batches))
            start_point = batch*BATCH_SIZE
            n_records = min(BATCH_SIZE, N_RECORDS - start_point)
            x0, y0 = load_and_preprocess_dataset(s=start_point, n=n_records)
            x_train, y_train = prepare_full_training_set(df_x=x0, df_y=y0,
                                                         number_of_samples_per_lenght=N_BATCH_SAMPLES,
                                                         number_of_sample_lenghts=N_LENGHTS)
            print('TRAINING')
            model.fit(x_train[batch], y_train[batch])       # WARNING!!! NOT SURE IF THE 'FIT' ALONE CAN BE USED TO DO INCREMENTAL TRAINING STEPS
        print('SAVING')
        filename, i_f = r'./nn/saved_models/RNN_06_test_0', 0
        while os.path.exists(filename):
            i_f += 1
            filename = r'./nn/saved_models/RNN_06_test_' + str(i_f)
        model.save(filename)
        print('PROCESS ENDED')

    if predict:
        model = keras.models.load_model(rnn_to_use)

        # PREPARE THE TEST SET:
        s_test, n_test = N_TRAIN, N_TEST
        x0, y0 = load_and_preprocess_dataset(s=s_test, n=n_test)
        x_test, y_test = prepare_full_test_set(df_x=x0, df_y=y0)

        # PREDICT
        y_pred = []
        print('prediction N#:')
        for i in range(len(x_test)):
            y_pred.append(model.predict(x_test[i]))
            print(i, end='\r')
        m = keras.metrics.MeanSquaredError()
        m.update_state(y_test, y_pred)
        outcome = m.result().numpy()
        print('mean squared error:', outcome)

        # ROLLING BACK THE PREDICTED VALUES TO 'RESP' FORM:
        x_originals, y_originals = load_data_set(start=s_test, n=n_test)
        y_hat_pos = [element.shape[1] + i - 1 for i, element in enumerate(x_test)]
        resps_original = np.asarray([y_originals.iloc[p]['resp'] for p in y_hat_pos])     # using a numpy mask instead of a generator?

        weights = np.asarray([z[0][-1][0] for z in x_test])
        y_used_for_test = np.asarray([z[0][0] for z in y_test])
        test_weights, y_inverted = convert_back_pred_y_hat_to_prob_hat_inputlike(w_df=weights,
                                                                                 y_df=y_used_for_test)
        y_used_for_pred = np.asarray([z[0][0] for z in y_pred])
        original_weights, y_pred_inverted = convert_back_pred_y_hat_to_prob_hat_inputlike(w_df=weights,
                                                                                          y_df=y_used_for_pred)
# WARNING: CHECK THE PROPER INCLUSION OF THE DATES!!
        # EVALUATING NOW THE KAGGLE SCORE ON THE SELF-TEST SET (= MAXIMUM ALLOWABLE SCORE)
        k_score = evaluate_kaggle_score(w=test_weights, y_inv=y_inverted)

        if plot_consistency_check:
            from matplotlib import pyplot as plt
            plt.figure()
            plt.plot(y_used_for_test, label='TEST SET y_hat (transformed)')
            plt.plot(y_inverted, label='TEST SET y_hat (originals)')
            plt.plot(y_used_for_pred, label='PREDICTED y_hat (transformed)')
            plt.plot(y_pred_inverted, label='PREDICTED y_hat (inverted)')
            plt.legend()
            plt.show()
