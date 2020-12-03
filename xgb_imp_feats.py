from sklearn.metrics import accuracy_score
import pandas as pd
import xgboost as xgb
import pickle
import gc
from tqdm import tqdm

## walk forward and trying to find stable features


train = pd.read_csv('input/train.csv')
## TODO not sure if it's the right thing to do
## very unbalance if include 0 weight, maybe use imblearn.RandomSampler
train = train[train['weight'] != 0]
train['action'] = (train.weight*train.resp>0).astype('int')


window = 100
epochs =  range(window, 500 - window + 1, window )

feat_importance = {}
for idt in tqdm(epochs):

    train_fold = train[(train['date'] >= idt - window) & (train['date'] < idt)] ##
    test_fold = train[(train['date'] >= idt) & (train['date'] < idt + window)]

    train_X, train_y = train_fold[list(map(lambda x: f"feature_{x}", range(130))) + ['weight']].fillna(-999), \
                        train_fold['action']

    test_X, test_y = test_fold[list(map(lambda x: f"feature_{x}", range(130))) + ['weight']].fillna(-999), \
                       test_fold['action']

    del train_fold

    train_weight = train_X.pop('weight')
    _ = test_X.pop('weight')

    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=12,
        learning_rate=0.5,
        subsample=0.76,
        colsample_bytree=0.52,
        random_state=666
    )

    ## maybe try get rid of the outlier weight of 126?
    clf.fit(train_X, train_y, sample_weight=train_weight)

    ## accuracy score
    print(f"test accuracy for [{idt}:{idt+window}] :{accuracy_score(test_y, clf.predict(test_X))}")

    importance_df = clf.get_booster().get_score(importance_type='weight')

    del clf
    gc.collect()

    features_impt_xgb = pd.DataFrame(list(importance_df.items()), columns = ['FEATURE' , 'SCORE']).sort_values('SCORE', ascending = False)

    feat_importance[idt] = features_impt_xgb



pickle.dump(feat_importance, open('output/feat_imp/wlk_fwd_feat_imp.pkl', "wb" ))







