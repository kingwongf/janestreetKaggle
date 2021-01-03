# coding=utf-8
import numpy as np
import janestreetKaggle.tools.big_csv_reader as bcr
import tensorflow as tf
from tensorflow import keras

# JUST A COPY OF TEST_05 FOR NOW. TO BE DEVELOPED

# THIS IS A SIMPLE LEARNING THE NEXT NUMBER IN A SEQUENCE OF 5 INTEGERS

# GENERATION OF AN RNN WITH n.1 LSTM, WORKING WITH A "WORD-VOCABULARY" OF 5 VECTORS. THE RANGE OF NUMBERS IS BETWEEN 0 AND 9
# EACH NUMBER IS ENCODED AS A 5-ROWS HOT VECTOR
# EACH SEQUENCE IS A SEQUENCE OF N.5 "10-ROW-VECTORS"


N_TRAIN = 10000
N_TEST = 100


def generate_hot_vectors(array: np.array) -> np.array:
    return tf.keras.utils.to_categorical(array, num_classes=10)


def generate_sequences(n=N_TRAIN):
    np.random.seed(3)
    start = np.random.randint(10, size=n)
    sequence = np.zeros((n, 5))
    for i, x in enumerate(start):
        sequence[i] = np.arange(x, x + 5) % 10
    return sequence.astype(int)


# GENERATE TRAIN SET
raw_inputs = generate_sequences()
inputs = generate_hot_vectors(raw_inputs)
raw_y_hats = ((raw_inputs[:, -1] + 1) % 10)
y_hats = generate_hot_vectors(raw_y_hats)

# GENERATE TEST SET
raw_t_inputs = generate_sequences(n=N_TEST)
t_inputs = generate_hot_vectors(raw_t_inputs)
raw_t_y_hats = ((raw_t_inputs[:, -1] + 1) % 10)
t_y_hats = generate_hot_vectors(raw_t_y_hats)

# BUILD THE MODEL (here each input is a 5-element sequence. Next step: feed variable-lenght sequences)
model_1 = keras.Sequential()
model_1.add(keras.layers.LSTM(64, input_shape=(5, 10)))
model_1.add(keras.layers.Dense(10, activation='softmax'))
model_1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_1.fit(inputs, y_hats, epochs=3)

# PREDICT AND EVALUATE ON THE TEST SET
r = model_1.predict(t_inputs)
result = r.argmax(axis=1)
n_right_answers = (raw_t_y_hats == result).sum()/N_TEST
print('accuracy on test set: %.2f %%' % (n_right_answers*100))
