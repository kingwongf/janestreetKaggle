import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import pickle

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
        # tree_method='gpu_hist'
    )
    return model


def objective(trial):
    model = create_model(trial)

    ## TODO add sample weight here
    model.fit(X_train, y_train)
    score = accuracy_score(
        y_train,
        model.predict(X_train)
    )
    return score



tags_pcas = pickle.load(open("output/tags_pcas.pkl", "rb"))

for tag, (tag_pca, x_pca, y_resampled) in tags_pcas.items():
    X_train = x_pca
    y_train = y_resampled



    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=70)
    params = study.best_params
    params['random_state'] = 666
    params['tree_method'] = 'gpu_hist'

    print(params)