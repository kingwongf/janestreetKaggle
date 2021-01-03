# coding=utf-8
import numpy as np
import janestreetKaggle.tools.big_csv_reader as bcr
from tensorflow import keras

# JUST A COPY OF TEST_05 FOR NOW. TO BE DEVELOPED

# THIS IS A SIMPLE LEARNING THE NEXT NUMBER IN A SEQUENCE OF 5 INTEGERS

# GENERATION OF AN RNN WITH n.1 LSTM, WORKING WITH A "WORD-VOCABULARY" OF 5 VECTORS. THE RANGE OF NUMBERS IS BETWEEN 0 AND 9
# EACH NUMBER IS ENCODED AS A 5-ROWS HOT VECTOR
# EACH SEQUENCE IS A SEQUENCE OF N.5 "10-ROW-VECTORS"


N_TRAIN = 5000
N_TEST = 1000

# LOAD PART OF KAGGLE DATA
data = bcr.read_big_csv(filepath=bcr.file, n=N_TRAIN + N_TEST)


# DROP NAN COLUMNS
nan_col = [0, 7, 8, 11, 12, 17, 18, 21, 22, 27, 28, 31, 31, 55, 72, 74, 78, 80, 84, 86, 90, 92, 96, 98, 102,
           104, 108, 110, 114, 116, 120, 121]
nan_columns = ['date'] + ['feature_' + str(x) for x in nan_col]
data = data.drop(columns=nan_columns, axis=1)
nc = data.shape[1]

# GENERATE TRAIN SET:
# TRANSFORM NOW EACH ROW IN A "(138-5)LENGHT_ONE-HOT-VECTOR" (maybe suppress "weight=0"-rows?)
# 1. timesteps will be 500 for each input.
# 2. gap between each input will be 50
# 3. number of samples will be 20000/50 = 400
# 4. output will be softmax, containing resp_{1, 2, 3, 4} and resp, each one multiplied by the weight
# so, the input shape to train the RNN will be: shape=(400, 500, 133)
# let's try first with n=2000 -> so: shape = (40, 500, 133)
c = list(data.columns)
y_columns = ['resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4']
x_columns = sorted(list(set(c) - set(y_columns)))
y_data = data[y_columns].multiply(data['weight'], axis='index')
x_data = data[x_columns]
# delete now raw data to save memory
training_sequence = np.zeros((40, 500-1, nc-5))
training_y_hat = np.zeros((40, 5))
for i in range(40):
    print('sequence', i)
    current_sequence = x_data.iloc[40*i: 40*i+500-1]
    current_resp = y_data.iloc[40*i+500]
    training_sequence[i] = current_sequence
    training_y_hat[i] = current_resp
# add variable lenght sampling procedure
# remove Nan in a better way
# try now to mirror the score into the metric: as close as possible...
# find max and min in the features
# find correlation in the features and extract the only one useful
# find how many nans the feature data have

# GENERATE THE TEST SET
test_sequence = np.zeros((10, 500-1, nc-5))
test_y_hat = np.zeros((10, 5))
for i in range(10):
    print('sequence', i)
    current_sequence = x_data.iloc[N_TEST + 10*i: N_TEST + 10*i+500-1]
    current_resp = y_data.iloc[N_TEST + 10*i+500]
    test_sequence[i] = current_sequence
    test_y_hat[i] = current_resp

# BUILD THE MODEL (here each input is a 5-element sequence. Next step: feed variable-lenght sequences)
model_1 = keras.Sequential()
model_1.add(keras.layers.LSTM(128, input_shape=(500, nc-5)))
model_1.add(keras.layers.Dense(5, activation='softmax'))
model_1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_1.fit(training_sequence, training_y_hat, epochs=3)
model_1.summary()

# I need to insert the one-hot vector encoding for the result?
# I NEED TO NORMALIZE FIRST THE FEATURES, TURNING TO 1-HOT-VECTORS WHERE EACH VECTOR-COMPONENT IS BETWEEN 0 AND 1
# FIRST STRATEGY: encode each resp{1..5} to a one-hot-vector (many dimensions!!) and then have a softmax
# SECOND STRATEGY(BEST?): have a sigmoid for each of the 5 output neurons, and encode the resp{1..5} to an interval [0..1]
# In fact, maybe I don't need a softmax: just 5 neurons with a sigmoid


# PREDICT AND EVALUATE ON THE TEST SET
r = model_1.predict(test_sequence)
# result = r.argmax(axis=1)
# n_right_answers = (raw_t_y_hats == result).sum()/N_TEST
# print('accuracy on test set: %.2f %%' % (n_right_answers*100))
