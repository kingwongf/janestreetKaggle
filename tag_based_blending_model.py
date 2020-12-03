import pickle
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from sklearn.metrics import accuracy_score
## tag_cols
with open('output/tag_blends/tagcols.pkl', 'rb') as f:
    tag_cols = pickle.load(f)


## load data
train = pd.read_csv('input/train.csv')
train = train[train['weight'] != 0]
train['action'] = (train.weight*train.resp>0).astype('int')


tag_feats = {}
for tag, tag_cols in tqdm(tag_cols.items()):
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.01,
        subsample=0.80,
        random_state=666,
        missing=-999
    )

    ## test of last 60 days
    ## train everything before

    split_idt = 499-60
    model.fit(train[train['date']<split_idt][tag_cols], train[train['date']<split_idt]['action'])
    print(tag)
    print(accuracy_score( train[train['date']>=split_idt]['action'] , model.predict(train[train['date']>=split_idt][tag_cols])))

    importance_df = model.get_booster().get_score(importance_type='weight')
    tag_feats[tag] = pd.DataFrame(list(importance_df.items()), columns=['FEATURE', 'SCORE']).sort_values('SCORE',
                                                                                                            ascending=False)

    del model

with open('output/tag_blends/tagcols.pkl', 'wb') as f:
    pickle.dump(tag_feats, f)