# coding=utf-8
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as k
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from hyperopt import hp, fmin, tpe, Trials

warnings.filterwarnings("ignore")
print('Tensorflow version:', tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE


def set_up_hardware():
    try:
        # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None
    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        my_strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        # default distribution strategy in Tensorflow. Works on CPU and single GPU.
        my_strategy = tf.distribute.get_strategy()
    print("REPLICAS: ", strategy.num_replicas_in_sync)
    mixed_precision = False
    xla_accelerate = True
    if mixed_precision:
        from tensorflow.keras.mixed_precision import experimental as mixed_precision
        if tpu:
            policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
        else:
            policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
        print('Mixed precision enabled')
    if xla_accelerate:
        tf.config.optimizer.set_jit(True)
        print('Accelerated Linear Algebra enabled')
    return my_strategy


def create_model(num_columns, num_labels, hidden_units, dropout_rates, label_smoothing, learning_rate):
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


def optimise(params):
    n_splits = 5
    batch_size = params['batch_size']
    hu = [params[f'hidden_unit_{n_layer}'] for n_layer in range(1, 5)]  # At least 4 layers in best submission
    dropout_rates = [params[f'dropout_{n_}'] for n_ in range(0, 5)]  # Dropout rates
    for n_layer in range(5, 10):
        if params[f'hidden_unit_{n_layer}'] != 0:
            hu.append(params[f'hidden_unit_{n_layer}'])
            dropout_rates.append(params[f'dropout_{n_layer}'])
        else:
            break
    p = {'hidden_units': hu,
         'dropout_rate': dropout_rates,
         'label_smoothing': params['label_smoothing'],
         'learning_rate': params['learning_rate']
         }
    oof = np.zeros(len(train['action']))
    gkf = GroupKFold(n_splits=n_splits)
    val_scores = []
    for fold, (tr, te) in enumerate(gkf.split(train['action'].values, train['action'].values, train['date'].values)):
        ckp_path = f'JSModel_{fold}.hdf5'
        x_tr, x_val = train.loc[tr, features].values, train.loc[te, features].values
        y_tr, y_val = train.loc[tr, 'action'].values, train.loc[te, 'action'].values
        with strategy.scope():
            model = create_model(num_columns=x_tr.shape[1],
                                 num_labels=1,
                                 hidden_units=p['hidden_units'],
                                 dropout_rates=p['dropout_rate'],
                                 label_smoothing=p['label_smoothing'],
                                 learning_rate=p['learning_rate'])
        rlr = ReduceLROnPlateau(monitor='val_AUC', factor=0.1, patience=3,
                                verbose=0, epsilon=1e-4, mode='max')
        ckp = ModelCheckpoint(ckp_path, monitor='val_AUC', verbose=0,
                              save_best_only=True, save_weights_only=True, mode='max')
        es = EarlyStopping(monitor='val_AUC', min_delta=0.0001, patience=7, mode='max',
                           baseline=None, restore_best_weights=True, verbose=0)
        model.fit(x_tr, y_tr, validation_data=(x_val, y_val),
                  epochs=1000, batch_size=batch_size,
                  callbacks=[rlr, ckp, es], verbose=0)
        oof[te] += model.predict(x_val, batch_size=batch_size * 4).ravel()  # added from best submission
        score = roc_auc_score(y_val, oof[te])
        val_scores.append(score)
        k.clear_session()
        del model
    return 1 - np.mean(val_scores)


strategy = set_up_hardware()
train = pd.read_csv('input/train.csv')
features = [c for c in train.columns if 'feature' in c]
print('Filling...')
f_mean = train[features[1:]].mean()
train = train[train.weight > 0].reset_index(drop=True)
train[features[1:]] = train[features[1:]].fillna(f_mean)
train['action'] = (train.resp > 0).astype('int')
print('Converting...')
f_mean = f_mean.values
np.save('f_mean.npy', f_mean)
print('Finish.')
param_space = {'hidden_unit_1': hp.choice('hidden_unit_1', [1152, 1024, 896, 768, 640, 512, 384, 256, 128]),
               'hidden_unit_2': hp.choice('hidden_unit_2', [1152, 1024, 896, 768, 640, 512, 384, 256, 128]),
               'hidden_unit_3': hp.choice('hidden_unit_3', [1152, 1024, 896, 768, 640, 512, 384, 256, 128]),
               'hidden_unit_4': hp.choice('hidden_unit_4', [1152, 1024, 896, 768, 640, 512, 384, 256, 128]),
               'hidden_unit_5': hp.choice('hidden_unit_5', [0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152]),
               'hidden_unit_6': hp.choice('hidden_unit_6', [0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152]),
               'hidden_unit_7': hp.choice('hidden_unit_7', [0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152]),
               'hidden_unit_8': hp.choice('hidden_unit_8', [0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152]),
               'hidden_unit_9': hp.choice('hidden_unit_9', [0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152]),
               'dropout_0': hp.uniform('dropout_0', 0, 0.5),
               'dropout_1': hp.uniform('dropout_1', 0, 0.5),
               'dropout_2': hp.uniform('dropout_2', 0, 0.5),
               'dropout_3': hp.uniform('dropout_3', 0, 0.5),
               'dropout_4': hp.uniform('dropout_4', 0, 0.5),
               'dropout_5': hp.uniform('dropout_5', 0, 0.5),
               'dropout_6': hp.uniform('dropout_6', 0, 0.5),
               'dropout_7': hp.uniform('dropout_7', 0, 0.5),
               'dropout_8': hp.uniform('dropout_8', 0, 0.5),
               'dropout_9': hp.uniform('dropout_9', 0, 0.5),
               'label_smoothing': hp.uniform('label_smoothing', 0, 0.1),
               'learning_rate': hp.uniform('learning_rate', 0, 0.10),
               'batch_size': hp.choice('batch_size', [4096, 4096 * 2])
               }
trials = Trials()
hopt = fmin(fn=optimise,
            space=param_space,
            algo=tpe.suggest,
            max_evals=10000,
            timeout=15 * 60 * 60,
            trials=trials,
            )
print(hopt)
