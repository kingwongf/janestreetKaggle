# coding=utf-8
import warnings
import gc
# import cudf
import pandas as pd
import numpy as np
# import cupy as cp
import janestreet
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from tqdm.notebook import tqdm
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

warnings.filterwarnings('ignore')
tf.random.set_seed(42)


def create_mlp(num_columns, num_labels, hidden_units, dropout_rates, label_smoothing, learning_rate):
    inp = tf.keras.layers.Input(shape=(num_columns,))
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dropout(dropout_rates[0])(x)
    for i in range(len(hidden_units)):
        x = tf.keras.layers.Dense(hidden_units[i])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)
        x = tf.keras.layers.Dropout(dropout_rates[i + 1])(x)

    x = tf.keras.layers.Dense(num_labels)(x)
    out = tf.keras.layers.Activation('sigmoid')(x)

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
                  metrics=tf.keras.metrics.AUC(name='AUC'),
                  )

    return model


train = pd.read_csv('/kaggle/input/jane-street-market-prediction/train.csv')
features = [c for c in train.columns if 'feature' in c]
print('Filling...')
f_mean = train[features[1:]].mean()
train = train[train.weight > 0].reset_index(drop=True)
train[features[1:]] = train[features[1:]].fillna(f_mean)
train['action'] = (train.resp > 0).astype('int')
print('Converting...')
# train = train.to_pandas()
f_mean = f_mean.values
np.save('f_mean.npy', f_mean)
print('Finish.')

batch_size = 4096
hidden_units = [384, 896, 896, 384]
dropout_rates = [0.10143786981358652, 0.19720339053599725, 0.2703017847244654, 0.23148340929571917, 0.2357768967777311]
label_smoothing = 1e-2
learning_rate = 1e-3
oof = np.zeros(len(train['action']))
gkf = GroupKFold(n_splits=5)
for fold, (tr, te) in enumerate(gkf.split(train['action'].values, train['action'].values, train['date'].values)):
    X_tr, X_val = train.loc[tr, features].values, train.loc[te, features].values
    y_tr, y_val = train.loc[tr, 'action'].values, train.loc[te, 'action'].values
    ckp_path = f'JSModel_{fold}.hdf5'
    model = create_mlp(X_tr.shape[1],
                       1, hidden_units, dropout_rates, label_smoothing, learning_rate)
    rlr = ReduceLROnPlateau(monitor='val_AUC', factor=0.1, patience=3, verbose=0,
                            min_delta=1e-4, mode='max')
    ckp = ModelCheckpoint(ckp_path, monitor='val_AUC', verbose=0,
                          save_best_only=True, save_weights_only=True, mode='max')
    es = EarlyStopping(monitor='val_AUC', min_delta=1e-4, patience=7, mode='max',
                       baseline=None, restore_best_weights=True, verbose=0)
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=1000,
              batch_size=batch_size, callbacks=[rlr, ckp, es], verbose=0)
    oof[te] += model.predict(X_val, batch_size=batch_size * 4).ravel()
    score = roc_auc_score(y_val, oof[te])
    print(f'Fold {fold} ROC AUC:\t', score)
    # Finetune 3 epochs on validation set with small learning rate
    model = create_mlp(X_tr.shape[1], 1, hidden_units, dropout_rates, label_smoothing, learning_rate / 100)
    model.load_weights(ckp_path)
    model.fit(X_val, y_val, epochs=3, batch_size=batch_size, verbose=0)
    model.save_weights(ckp_path)
    K.clear_session()
    del model
    rubbish = gc.collect()
num_models = 2
models = []
for i in range(num_models):
    clf = create_mlp(len(features), 1, hidden_units, dropout_rates, label_smoothing, learning_rate)
    clf.load_weights(f'./JSModel_{i}.hdf5')
    #     clf.load_weights(f'./JSModel_{i}.hdf5')
    models.append(clf)
f_mean = np.load('./f_mean.npy')
env = janestreet.make_env()
env_iter = env.iter_test()
opt_th = 0.5
for (test_df, pred_df) in tqdm(env_iter):
    if test_df['weight'].item() > 0:
        x_tt = test_df.loc[:, features].values
        if np.isnan(x_tt[:, 1:].sum()):
            x_tt[:, 1:] = np.nan_to_num(x_tt[:, 1:]) + np.isnan(x_tt[:, 1:]) * f_mean
        pred = 0.
        for clf in models:
            pred += clf(x_tt, training=False).numpy().item() / num_models
        #         pred = models[0](x_tt, training = False).numpy().item()
        pred_df.action = np.where(pred >= opt_th, 1, 0).astype(int)
    else:
        pred_df.action = 0
    env.predict(pred_df)
param_space = {'hidden_unit_1': [4096, 2048, 1024, 512],
               'hidden_unit_2': [4096, 2048, 1024, 512],
               'hidden_unit_3': [4096, 2048, 1024, 512, 0],
               'hidden_unit_4': [4096, 2048, 1024, 512, 0],
               'hidden_unit_5': [4096, 2048, 1024, 512, 0],
               'hidden_unit_6': [4096, 2048, 1024, 512, 0],
               'hidden_unit_7': [4096, 2048, 1024, 512, 0],
               'hidden_unit_8': [4096, 2048, 1024, 512, 0],
               'hidden_unit_9': [4096, 2048, 1024, 512, 0],
               'hidden_unit_10': [4096, 2048, 1024, 512, 0]
               }
hu = [param_space[f'hidden_unit_{n_layer}'] for n_layer in range(1, 5)]
for n_layer in range(5, 11):
    if param_space[f'hidden_unit_{n_layer}'] == 0:
        break
    else:
        hu.append(param_space[f'hidden_unit_{n_layer}'])
