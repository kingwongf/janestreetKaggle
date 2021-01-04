# todo: correlate the results (resp) to the very high_variance_features
# todo: how to normalize data that will come in the future? Which will be then minima and maxima?

# coding=utf-8
import numpy as np
import pandas as pd
import janestreetKaggle.tools.big_csv_reader as bcr
# from tensorflow import keras


N_TRAIN = 500
N_TEST = 100
N_RECORDS = 2390490
MINIMUM_SEQUENCE_LENGHT, MAXIMUM_SEQUENCE_LENGHT = (10, 30)

y_columns = ['resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4']


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


def normalise_data(df: pd.DataFrame):
    # (review lessons). Also: MAYBE WORTH TO AVOID TURNING "-1" IN FEATURE_0 TO "0"
    stats = pd.read_csv(r'C:\Kaggle-King\janestreetKaggle\tools\feature_stats.csv',
                        index_col=0)        # load the file with the minima & maxima
    for c in df.columns:    # normalise each column, except for weights and resps (map between 0 and 1)
        x_min, x_max = stats.loc[c]['minima'], stats.loc[c]['maxima']
        a = 1/(x_max - x_min)
        b = -x_min*a
        df[c] = a*df[c] + b
    return df


def prepare_one_hot_vectors_input_set(df_x: pd.DataFrame, df_y: pd.DataFrame, sample_size: int, start: int):
    one_training_sample = df_x.iloc[start: start + sample_size].to_numpy()
    y_hat = df_y.iloc[start + sample_size].to_numpy()
    return one_training_sample, y_hat


def prepare_hot_vectors_input_set(df_x: pd.DataFrame, df_y: pd.DataFrame, start_points: np.array, sample_size: int):
    number_of_samples = len(start_points)
    x_training_set = np.zeros((number_of_samples, sample_size, df_x.shape[1]))
    y_training_set = np.zeros((number_of_samples, df_y.shape[1]))
    for i in range(number_of_samples):
        one_input_sequence, y_hat = prepare_one_hot_vectors_input_set(df_x=df_x, df_y=df_y,
                                                                      sample_size=sample_size, start=start_points[i])
        x_training_set[i], y_training_set[i] = one_input_sequence, y_hat
    return x_training_set, y_training_set


def prepare_full_training_set(df_x: pd.DataFrame, df_y: pd.DataFrame,
                              number_of_samples_per_lenght: int, number_of_sample_lenghts: int):
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


# HERE! IMPLEMENT THE SCORE METRIC NOW!
def generate_the_score_metric(data):
    # THIS IS ANOTHER DIFFICULT
    y_data = data[y_columns].multiply(data['weight'], axis='index')
    return


def build_the_model(training_sequence, training_y_hat):
    # AMEND ACCORDINGLY THE NEW ARCHITECTURE
    model_1 = keras.Sequential()
    model_1.add(keras.layers.LSTM(128, input_shape=(500, nc - 5)))
    model_1.add(keras.layers.Dense(5, activation='softmax'))
    model_1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_1.fit(training_sequence, training_y_hat, epochs=3)
    model_1.summary()
    return


def train_the_model_with_varaiable_lenght_sequences():
    return


def predict_and_evaluate(model_1, test_sequence):
    r = model_1.predict(test_sequence)
    # result = r.argmax(axis=1)
    # n_right_answers = (raw_t_y_hats == result).sum()/N_TEST
    # print('accuracy on test set: %.2f %%' % (n_right_answers*100))


# TRANSFORM NOW EACH ROW IN A "(138-5)LENGHT_ONE-HOT-VECTOR" (maybe suppress "weight=0"-rows?)
# 1. timesteps will be 500 for each input.
# 2. gap between each input will be 50
# 3. number of samples will be 20000/50 = 400
# 4. output will be softmax, containing resp_{1, 2, 3, 4} and resp, each one multiplied by the weight
# so, the input shape to train the RNN will be: shape=(400, 500, 133)
# let's try first with n=2000 -> so: shape = (40, 500, 133)

# delete now raw data to save memory

# add variable lenght sampling procedure
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
    x, y = load_data_set()
    x = suppress_correlated_features(x)
    x = fill_nan(x)
    x, y = normalise_data(x), normalise_data(y)
    x_training = prepare_full_training_set(df_x=x, df_y=y, number_of_samples_per_lenght=20, number_of_sample_lenghts=4)
    print('ok')
