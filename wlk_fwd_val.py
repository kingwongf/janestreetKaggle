import pandas as pd
import datatable as dt
from sklearn.decomposition import PCA
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score
import xgboost as xgb

## spliting into times series k fold with date column, which starts from 0 ends 499
## split into 50 folds

train = dt.fread('input/train.csv')
train = train.to_pandas()
## TODO not sure if it's the right thing to do
train = train[train['weight'] != 0]


train['action'] = (train.weight*train.resp>0).astype('int')


sampler = TPESampler(seed=666)



def create_model(trial):
    max_depth = trial.suggest_int("max_depth", 2, 12)
    n_estimators = trial.suggest_int("n_estimators", 2, 600)
    learning_rate = trial.suggest_uniform('learning_rate', 0.0001, 0.99)
    subsample = trial.suggest_uniform('subsample', 0.0001, 1.0)
    colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.0000001, 1)

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=666,
        tree_method='gpu_hist'
    )
    return model


def objective(trial):
    model = create_model(trial)

    ## trying walk-forward with rolling window for now
    score=0
    window = 100
    epochs =  range(window, 500 - window + 1)
    for idt in epochs:

        train_fold = train[(train['date'] >= idt - window) & (train['date'] < idt)]
        test_fold = train[(train['date'] >= idt) & (train['date'] < idt + window)]

        print(f"train date : {train_fold['date'].min()} to {train_fold['date'].max()}")
        print(f"test date : {test_fold['date'].min()} to {test_fold['date'].max()}")

        ## train_X, train_y,

        train_X, train_y = train_fold[list(map(lambda x: f"feature_{x}", range(130))) + ['weight']].fillna(-999), \
                           train_fold['action']
        test_X, test_y = test_fold[list(map(lambda x: f"feature_{x}", range(130))) + ['weight']].fillna(-999), \
                         test_fold['action']

        train_weight, test_weight = train_X.pop('weight'), test_X.pop('weight')

        ## TODO put Feature Engineering
        pca = PCA(n_components=80)
        pca.fit(train_X)
        pca_train_X, pca_test_X = pca.transform(train_X), pca.transform(test_X)

        ## TODO add sample weight here
        model.fit(pca_train_X, train_y, sample_weight=train_weight)
        score += accuracy_score(test_y,model.predict(pca_test_X))

    return score/len(epochs)



study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=70)
params = study.best_params
params['random_state'] = 666
params['tree_method'] = 'gpu_hist'
print(params)


