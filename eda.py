import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
import xgboost as xgb
import imblearn
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA


## Can't do this because we don't know which resp_ij associated with each row
# def utility_score(weight, resp, action, num_i):
#     '''
#
#     :param weight: braodcast to shape (1,4)
#     :param resp: should have resp_1 to resp_4,
#     :param action: braodcast to shape (1,4)
#     :param num_i: number of unique dates in the test set
#     :return:
#     '''
#     ## should have shape (num_i,1)
#     p_i = np.sum(weight*resp*action)
#     t = (np.sum(p_i) / np.sqrt(np.sum(p_i**2))) * np.sqrt(250 / num_i)
#     return min(max(t, 0), 6) * np.sum(p_i)



train = pd.read_csv('input/train.csv')
feat_meta = pd.read_csv('input/features.csv')

print ("Data is loaded!")




train = train[train['weight'] != 0]

train['action'] = (train.weight*train.resp>0).astype('int')


X_train = train[list(map(lambda x: f"feature_{x}", range(130)))+ ['weight']]
y_train = train.loc[:, 'action']

X_train = X_train.fillna(-999)






## TODO 0. Fit individual models based on feature meta data

tags_cols = {}
for tag_col in feat_meta.drop('feature', axis=1).columns:
    tags_cols[tag_col] = feat_meta[feat_meta[tag_col]==True]['feature'].tolist()


## TODO: 1.What if I only train last 60 days, will there be a better model?/  seggregate into 60 days windows based on date columns/


## TODO: Better way to resample? Should use the 'Weight' column
cc = RandomUnderSampler(random_state=0)

X_resampled, y_resampled = cc.fit_resample(X_train , y_train)
train_resampled_weight = X_resampled['weight']

X_resampled.drop('weight', axis=1, inplace=True)

del X_train, y_train


## TODO: 2. PCA features and take first 50, 100 etc.
tags_pcas = {}
for tag, tag_cols in tags_cols.items():
    tag_pca = PCA(n_components=int(0.5*len(tag_cols)))
    tag_pca.fit(X_resampled[tag_cols])
    x_pca = tag_pca.transform(X_resampled[tag_cols])
    tags_pcas[tag] = (tag_pca, x_pca)

del X_resampled

## TODO: 3. train multiple models and predict based on which window
## TODO using the weights column to train XGB
tag_collections = {}
## should have 1. feature columns used in each tag, 2. pca in each tag, 3. model used in each tag
for tag, (tags_pca, pca_x) in tags_pcas.items():
    tag_collections[tag] = (tags_cols[tag],
                            tags_pcas[tag],
                            xgb.XGBClassifier(base_score=0.5,
                                              max_depth=8, learning_rate=0.05,
                                              booster='gbtree', colsample_bylevel=1,
                                              colsample_bynode=1, colsample_bytree=0.7,
                                              gamma=0, gpu_id=0,
                                              importance_type='gain', interaction_constraints='',
                                              n_estimators=500, missing=-999,
                                              random_state=1964,
                                              n_jobs=0, num_parallel_tree=1,
                                              objective='binary:logistic').fit(
                            pca_x, y_resampled, sample_weight=train_resampled_weight),
                            )

pickle.dump(tag_collections, open( "tag_collections.pkl", "wb" ) )


## TODO need of time series CV to optimise params
## TODO: Need to retrain model after every test_df
from input import janestreet

env = janestreet.make_env()
iter_test = env.iter_test()
for (test_df, sample_prediction_df) in iter_test:
    test_df = test_df.fillna(-999).loc[:, test_df.columns.str.contains('feature')]
    test_preds = np.array([])
    for tag, (tag_cols, tag_pca, tag_xgb) in tag_collections.items():
        test_preds = np.concatenate([test_preds, tag_xgb.predict(tag_pca.transform(test_df[tag_cols]))], axis=1)
    sample_prediction_df.action = 1 if test_preds.mean() > 1 else 0
    env.predict(sample_prediction_df)
