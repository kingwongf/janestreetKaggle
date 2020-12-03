from sklearn.metrics import accuracy_score
import pandas as pd
import xgboost as xgb
import pickle
import gc
from tqdm import tqdm
train = pd.read_csv('input/train.csv')
## TODO not sure if it's the right thing to do
## very unbalance if include 0 weight, maybe use imblearn.RandomSampler
train = train[train['weight'] != 0]
train['action'] = (train.weight*train.resp>0).astype('int')


window = 50
epochs =  range(window, 500 - window + 1, 50 )

for idt in tqdm(epochs):

    train_fold = train[(train['date'] < idt)] ## (train['date'] >= idt - window) &
    test_fold = train[(train['date'] >= idt) & (train['date'] < idt + window)]

    train_X, train_y = train_fold[list(map(lambda x: f"feature_{x}", range(130))) + ['weight']].fillna(-999), \
                        train_fold['action']

    test_X, test_y = test_fold[list(map(lambda x: f"feature_{x}", range(130))) + ['weight']].fillna(-999), \
                       test_fold['action']

    pickle.dump({'train_X': train_X, 'test_X': test_X,'train_y':  train_y, 'test_y':test_y},
                open(f'output/wlk_fwd_splt/{idt}.pkl', "wb" ))