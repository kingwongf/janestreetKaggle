import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import xgboost as xgb
from sklearn.metrics import accuracy_score

with open('output/feat_imp/wlk_fwd_feat_imp.pkl', "rb" ) as file:
    wlk_fwd_feat_imp = pickle.load(file)


feat_df = pd.DataFrame([feat_df['FEATURE'].rename(epoch) for epoch, feat_df in wlk_fwd_feat_imp.items()]).T

feat_li = np.unique(feat_df.head(20).values.flatten())

window = 100
epochs = range(window, 500 - window + 1, window )

train = pd.read_csv('input/train.csv')
train = train[train['weight'] != 0]
train['action'] = (train.weight*train.resp>0).astype('int')


feat_importance = {}
for idt in tqdm(epochs):
    train_fold = train[(train['date'] < idt)]  ## (train['date'] >= idt - window) &
    test_fold = train[(train['date'] >= idt) & (train['date'] < idt + window)]

    train_X, train_y = train_fold[feat_li].fillna(-999), train_fold['action']

    train_weight = train_fold['weight']

    test_X, test_y = test_fold[feat_li].fillna(-999), test_fold['action']

    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.5,
        subsample=0.76,
        colsample_bytree=0.52,
        random_state=666,
        missing=-999
    )

    ## maybe try get rid of the outlier weight of 126?

    ## TODO test if sample_weight helps?
    clf.fit(train_X, train_y, sample_weight=train_weight)

    ## accuracy score
    print(f"test accuracy for [{idt}:{idt + window}] :{accuracy_score(test_y, clf.predict(test_X))}")