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


## TODO need of cross-valiation to hyperparamter runing

## spliting into times series k fold with date column, which starts from 0 ends 499
## split into 50 folds

# k_folds = {}
# for dt in range(10,500):
#     fold = train[(train['date'] == dt - 10)|(train['date'] == dt)]
#     ## X, y
#
#     k_folds[dt] = fold[list(map(lambda x: f"feature_{x}", range(130)))+ ['weight']], fold.loc[:, 'action']




## TODO 0. Fit individual models based on feature meta data

tags_cols = {}
for tag_col in feat_meta.drop('feature', axis=1).columns:
    tags_cols[tag_col] = feat_meta[feat_meta[tag_col]==True]['feature'].tolist()

with open('output/tag_blends/tagcols.pkl', 'wb') as f:
    pickle.dump(tags_cols, f)


## TODO: 1.What if I only train last 60 days, will there be a better model?/  seggregate into 60 days windows based on date columns/


## TODO: Better way to resample? Should use the 'Weight' column
cc = RandomUnderSampler(random_state=0)

X_resampled, y_resampled = cc.fit_resample(X_train , y_train)
train_resampled_weight = X_resampled['weight']

X_resampled.drop('weight', axis=1, inplace=True)

del X_train, y_train


## TODO: 2. PCA features and take first 50, 100 etc.
tags_pcas = {}
percentage_compoenets = 0.80

for tag, tag_cols in tags_cols.items():
    tag_pca = PCA(n_components=int(percentage_compoenets*len(tag_cols)))
    tag_pca.fit(X_resampled[tag_cols])
    x_pca = tag_pca.transform(X_resampled[tag_cols])
    tags_pcas[tag] = (tag_pca, x_pca, y_resampled)

del X_resampled


pickle.dump(tags_pcas, open( "output/tags_pcas.pkl", "wb" ) )


tags_pca = pickle.load(open( "output/tags_pcas.pkl", "rb" ))

