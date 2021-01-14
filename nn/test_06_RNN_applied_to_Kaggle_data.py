# todo: correlate the results (resp) to the very high_variance_features
# todo: how to normalize data that will come in the future? Which will be then minima and maxima?

# coding=utf-8
import numpy as np
import pandas as pd
import janestreetKaggle.tools.big_csv_reader as bcr
from tensorflow import keras
import os

test = True
if test:
    HOT_VECTOR_SIZE = 84
    BATCH_SIZE = 5000
    MINIMUM_SEQUENCE_LENGHT, MAXIMUM_SEQUENCE_LENGHT = (150, 500)
    N_BATCH_SAMPLES = 50
    N_LENGHTS = 10
    N_RECORDS = 6000
    N_TRAIN = 5000
    N_TEST = N_RECORDS - N_TRAIN
else:
    HOT_VECTOR_SIZE = 84
    BATCH_SIZE = 500000
    MINIMUM_SEQUENCE_LENGHT, MAXIMUM_SEQUENCE_LENGHT = (150, 500)
    N_BATCH_SAMPLES = 10000
    N_LENGHTS = 20
    N_RECORDS = 2390490
    N_TRAIN = 2300000
    N_TEST = N_RECORDS - N_TRAIN
# better dividing the training set in aliquot parts
y_columns = ['resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4']


def build_the_model():
    model_1 = keras.Sequential()
    model_1.add(keras.layers.LSTM(128, input_shape=(None, 84)))
    model_1.add(keras.layers.Dense(5, activation='sigmoid'))
    model_1.compile(loss='MeanSquaredError', optimizer='adam', metrics=['accuracy'])
    print(model_1.summary())
    return model_1


def convert_back_pred_y_hat_to_prob_hat_inputlike(w_df: np.array, y_df: np.array):
    w_stats = pd.read_csv(r'C:\Kaggle-King\janestreetKaggle\tools\weighted_responses_stats.csv', index_col=0)       # load the file with the wegighted minima & maxima
    w_resp_min, w_resp_max = w_stats.loc['resp']['minima'], w_stats.loc['resp']['maxima']
    a = 1 / (w_resp_max - w_resp_min)
    b = -w_resp_min * a
    w_original_resp = (y_df - b)/a
    # w_original_resp = y_df.apply(lambda z: z/a - b/a)

    stats = pd.read_csv(r'C:\Kaggle-King\janestreetKaggle\tools\feature_stats.csv', index_col=0)        # load the file with the minima & maxima
    w_min, w_max = stats.loc['weight']['minima'], stats.loc['weight']['maxima']
    a = 1 / (w_max - w_min)
    b = -w_min * a
    w_original = (w_df - b)/a
    original_resp = w_original_resp / w_original

    return w_original, original_resp


def load_data_set(start: int = None, n: int = N_TRAIN + N_TEST):
    data = bcr.read_big_csv(filepath=bcr.file, s=start, n=n)
    c = list(data.columns)
    x_columns = sorted(list(set(c) - set(y_columns)))
    y_data = data[y_columns]
    x_data = data[x_columns]
    return x_data, y_data


def _suppress_weight_0_rows():
    return


def _aggregate_features_in_tags():
    return


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


def _drop_nan_columns(df):
    nan_col = [0, 7, 8, 11, 12, 17, 18, 21, 22, 27, 28, 31, 31, 55, 72, 74, 78, 80, 84, 86, 90, 92, 96, 98, 102,
               104, 108, 110, 114, 116, 120, 121]
    nan_columns = ['date'] + ['feature_' + str(x) for x in nan_col]
    data = df.drop(columns=nan_columns, axis=1)
    nc = data.shape[1]
    return


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


def prepare_hot_vectors_input_set(df_x: pd.DataFrame, df_y: pd.DataFrame,
                                  start_points: np.array, sample_size: int,
                                  test_mode=False):
    number_of_samples = len(start_points)
    x_training_set = np.zeros((number_of_samples, sample_size, df_x.shape[1]))
    y_training_set = np.zeros((number_of_samples, df_y.shape[1]))
    for i in range(number_of_samples):
        _, one_input_sequence, y_hat = prepare_one_hot_vectors_input_set(df_x=df_x, df_y=df_y,
                                                                      sample_size=sample_size, start=start_points[i])
        x_training_set[i], y_training_set[i] = one_input_sequence, y_hat
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
    for i in sample_lenghts:
        start_points_segment = np.random.choice(range(len(df_x)-i),
                                                size=number_of_samples_per_lenght,
                                                replace=False)
        start_points.extend(start_points_segment)
    start_points = np.asarray(start_points)
    for i in range(number_of_sample_lenghts):
        start_points_chunk = start_points[number_of_samples_per_lenght*i: number_of_samples_per_lenght*(i+1)]
        x_training_chunk, y_training_chunk = prepare_hot_vectors_input_set(df_x=df_x, df_y=df_y, start_points=start_points_chunk, sample_size=sample_lenghts[i])
        full_x_training_set.append(x_training_chunk)
        full_y_training_set.append(y_training_chunk)
    return full_x_training_set, full_y_training_set


def prepare_full_test_set(df_x: pd.DataFrame, df_y: pd.DataFrame):
    print('ORGANISING DATA IN APPROPRIATE SETS... ')
    test_list_x, test_list_y = [], []
    test_size = len(df_x)
    for i in range(test_size - MINIMUM_SEQUENCE_LENGHT):
        sample_size = np.random.randint(MINIMUM_SEQUENCE_LENGHT, min(MAXIMUM_SEQUENCE_LENGHT, test_size - i))
        x__test, y__test = prepare_hot_vectors_input_set(df_x=df_x, df_y=df_y,
                                                         sample_size=sample_size,
                                                         start_points=[i])
        test_list_x.append(x__test)
        test_list_y.append(y__test)
    return test_list_x, test_list_y


def fully_prepare_data_sets(s, n):
    print('LOADING... ', end='')
    x, y = load_data_set(start=s, n=n)
    print('PURGING... ', end='')
    x = suppress_correlated_features(x)
    x = fill_nan(x)
    print('WEIGHTING Y HATS... ', end='')
    y = y.multiply(x['weight'], axis='index')  # weight all the resp and consider them as the effective y_hats
    print('NORMALISING NOW X AND Y_HATS')
    x = normalise_data(x, r'C:\Kaggle-King\janestreetKaggle\tools\feature_stats.csv')
    y = normalise_data(y, r'C:\Kaggle-King\janestreetKaggle\tools\weighted_responses_stats.csv')
    return x, y


def predict_and_evaluate(model_1, test_sequence):
    r = model_1.predict(test_sequence)
    # result = r.argmax(axis=1)
    # n_right_answers = (raw_t_y_hats == result).sum()/N_TEST
    # print('accuracy on test set: %.2f %%' % (n_right_answers*100))


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


# TRANSFORM NOW EACH ROW IN A "(138-5)LENGHT_ONE-HOT-VECTOR" (maybe suppress "weight=0"-rows?)
# 1. timesteps will be 500 for each input.
# 2. gap between each input will be 50
# 3. number of samples will be 20000/50 = 400
# 4. output will be softmax, containing resp_{1, 2, 3, 4} and resp, each one multiplied by the weight
# so, the input shape to train the RNN will be: shape=(400, 500, 133)
# let's try first with n=2000 -> so: shape = (40, 500, 133)

# delete now raw data to save memory

# add variable lenght sampling procedure - DONE
# remove Nan in a better way
# try now to mirror the score into the metric: as close as possible...
# find max and min in the features
# find correlation in the features and extract the only one useful
# find how many nans the feature data have

# I need to insert the one-hot vector encoding for the result?
# I NEED TO NORMALIZE FIRST THE FEATURES, TURNING TO 1-HOT-VECTORS WHERE EACH VECTOR-COMPONENT IS BETWEEN 0 AND 1
# FIRST STRATEGY: encode each resp{1..5} to a one-hot-vector (many dimensions!!) and then have a softmax
# SECOND STRATEGY(BEST?): have a sigmoid for each of the 5 output neurons, and encode the resp{1..5} to an interval [0..1]
# In fact, maybe I don't need a softmax: just 5 neurons with a sigmoid


if __name__ == '__main__':
    predict = True
    train = False
    inversion_test = True

    if train:
        np.random.seed(3)          # ensures the repeatability of the training
        model = build_the_model()
        number_of_batches = N_TRAIN // BATCH_SIZE + bool(N_TRAIN % BATCH_SIZE)
        for b in range(number_of_batches):
            print('TRAINING NOW BATCH %d / %d' % (b, number_of_batches))
            s = b*BATCH_SIZE
            n = min(BATCH_SIZE, N_RECORDS - s)
            x, y = fully_prepare_data_sets(s=s, n=n)
            x_train, y_train = prepare_full_training_set(df_x=x, df_y=y,
                                                         number_of_samples_per_lenght=N_BATCH_SAMPLES,
                                                         number_of_sample_lenghts=N_LENGHTS)
            print('TRAINING')
            model.fit(x_train[b], y_train[b])       # WARNING!!! IT SEEMS FIT ALONE CANNOT BE USED TO DO INCREMENTAL TRAINING STEPS
        print('SAVING')
        filename, i_f = r'./nn/saved_models/RNN_06_test_0', 0
        while os.path.exists(filename):
            i_f += 1
            filename = r'./nn/saved_models/RNN_06_test_' + str(i_f)
        model.save(filename)
        print('EVALUATING ON THE TEST SET ')
        # NOW TEST THE MODEL OVER THE TEST SET
        print('EVALUATING THE KAGGLE SCORE ON THE TRAIN AND TEST SETS ')
        print('PROCESS ENDED')

    if predict:
        model = keras.models.load_model(r'./nn/saved_models/RNN_06_test_2')
        # PREPARE THE TEST SET:
        s, n = 6000, 1000      # move 0 to 5000
        x, y = fully_prepare_data_sets(s=s, n=n)
        x_test, y_test = prepare_full_test_set(df_x=x, df_y=y)
        y_pred = []
        print('prediction N#:')
        for i in range(len(x_test)):
            y_pred.append(model.predict(x_test[i]))
            print(i, end='\r')
        m = keras.metrics.MeanSquaredError()
        m.update_state(y_test, y_pred)
        outcome = m.result().numpy()
        print('mean squared error:', outcome)
        # EVALUATE KAGGLE SCORE ON THE TEST SET:
        # the following can also probably be done more efficiently with Numpy (maybe using a mask)
        x_originals, y_originals = load_data_set(n=n)
        y_hat_pos = [element.shape[1]+i-1 for i, element in enumerate(x_test)]
        resps_original = np.asarray([y_originals.iloc[p]['resp'] for p in y_hat_pos])

        weights = np.asarray([z[0][-1][0] for z in x_test])
        y_used_for_test = np.asarray([z[0][0] for z in y_test])
        test_weights, y_inverted = convert_back_pred_y_hat_to_prob_hat_inputlike(w_df=weights,
                                                                                 y_df=y_used_for_test)
        y_used_for_pred = np.asarray([z[0][0] for z in y_pred])
        original_weights, y_pred_inverted = convert_back_pred_y_hat_to_prob_hat_inputlike(w_df=weights,
                                                                                          y_df=y_used_for_pred)
        # EVALUATING NOW THE KAGGLE SCORE ON THE SELF-TEST SET
        y_inverted[np.isneginf(y_inverted)] = 0
        y_inverted[np.isnan(y_inverted)] = 0
        probabilities = test_weights * y_inverted * (y_inverted > 0)
        probabilities_sum = probabilities.sum()
        probabilities_var = np.var(probabilities)
        unique_dates_number = len(set(x_originals['date']))
        t = probabilities_sum/probabilities_var * np.sqrt(250/unique_dates_number)
        u = min(max(t, 0), 6)*probabilities_sum

        # there is something strange here above: not quite the right back-algorithm... needs to be corrected.
        # Let's try to see what happens with y_test in place of y_pred
        # NEED FIRST TO CORRECT THE PREDICTION IN A WAY THAT, WHEN WEIGHT IS 0, THE PREDICTION IS 0.

        # 1. load all proper weights
        # 2. transform back y_pred: from [w * resps_{}] to [resps_{}] (only for the simple "RESP")
        # Do we need all that?? We can simply do, instead: new_resp (=weight*resp) * (reverted_prediction > 0)
        # first, compute the reverted prediction == 'DE-NORMALISE' data.
        # load from stats the resp min and max and invert the normalization.

# todo: the y_hats must be the probabilities. SO:
#  DONE - 1. built the "prob-y-hat-set", and normalise them to [0-1]
#  DONE - 2. train and evaluate against these y-hats
#  2b. resolve the problem of training the network in bits and pieces. Is it doing incrementally?
#  3. evaluate the kaggle-score on the test set: transform back y_hat to resps and also transform back weights. Do not forget to re-include the date.
#  4. evaluate the kaggle-score on the training set, according to the same methodology
#  5. introduce a dev set
#  6. introduce a final layer trasforming the ouputs back to "resp"


