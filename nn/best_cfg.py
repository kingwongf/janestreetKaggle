def load_data():
  return (scale_train_X, train_y), (si.transform(sc.transform(test_X)), test_y)
def get_model(n_feats):
  return keras.Sequential([
    keras.layers.Dense(32, input_shape=(n_feats,), activation=tf.nn.relu, kernel_regularizer='l1', name = "Dense_1"),
    keras.layers.BatchNormalization(),
    # keras.layers.Dropout(0.7,seed=2020),
    # keras.layers.Dense(64, activation=tf.nn.relu, kernel_regularizer='l1', name = "Dense_2"),
    # keras.layers.Dropout(0.1,seed=2020),
    # keras.layers.Dense(64, activation=tf.nn.relu, kernel_regularizer='l1', name = "Dense_3"),
    keras.layers.Dense(8, activation=tf.nn.relu, kernel_regularizer='l1', name = "Dense_4"),
    # keras.layers.Dense(1, activation=None, name = "logits"),
    # keras.layers.Dense(1, activation=tf.nn.softmax, name = "softmax")
    keras.layers.Dense(1, activation=tf.nn.sigmoid, name = "softmax")
  ])