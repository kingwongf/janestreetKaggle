import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
import imblearn
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats


def rm_outlier(df, threshold=5):
    '''

    :param df:
    :param threshold:
    :return:
    '''
    z = np.abs(stats.zscore(df, nan_policy='omit'))
    return df[(z < threshold).all(axis=1)].reset_index(drop=True)

def train_test_split(df, int_dt_window=100):
    '''
    Gives expanding train, test integer indices,
    '''
    epochs =  range(int_dt_window, max(df['date']) - int_dt_window + 1, int_dt_window )

    return [(df[(df['date'] < idt)].index.values,
             df[(df['date'] >= idt) &
                   (df['date'] < idt + int_dt_window)].index.values) for idt in epochs]

def train_pca(X, n_components):
    pca = PCA(n_components=n_components).fit(X)
    return (pca, pca.transform(X))

def norm_weight(train):
    train['weight'] = train['weight']/ train['weight'].sum()
    return train


def scaling(X, scaler='MinMaxScaler'):
    scaler_ = getattr(locals(), scaler)().fit(X)
    return (scaler, scaler.transform(X))

def utility(dates: np.ndarray, returns: np.ndarray,
            weights: np.ndarray, actions: np.ndarray) -> np.ndarray:

    gains = pd.DataFrame({
        'date': dates,
        'gain': weights * returns * actions
    })

    '''
	utility score of predictions of test dataset 
    :param dates: test dates in np.array (n,1)
    :param returns: test 'resp' in np.array (n,1)
    :param weights: test 'weight in np.array (n,1)
    :param actions: prediction of test's 'action' in np.array (n,1)
    :return: utility score of whole test dataset
    
    e.g.
    
    probs = clf.predict_proba( np.nan_to_num(sc.transform(test_X[feat_cols]), nan=-999) )[:,1] 
	thresholds = 0.5 + 0.20 * test_weight * (1/ max_weight)

	ind_tw = test_weight.astype(bool).astype(int)
	test_pred = (probs > thresholds).astype(int)*ind_tw

	print( utility( dates=train[train.date>=375]['date'], 
				   returns=train[train.date>=375]['resp'], 
				   weights=test_weight, actions=test_pred
				   )
			)
    '''


    p = gains.groupby('date').sum().to_numpy()

    i = len(set(dates))

    t = p.sum() / np.sqrt(np.power(p, 2).sum()) * np.sqrt(250 / i)

    return np.clip(t, a_min=0, a_max=6) * p.sum()


def mutualInfo(x,y,bXY,norm=False):
    cXY=np.histogram2d(x,y,bXY)[0] iXY=mutual_info_score(None,None,contingency=cXY) # mutual information if norm:
    hX=ss.entropy(np.histogram(x,bXY)[0]) # marginal hY=ss.entropy(np.histogram(y,bXY)[0]) # marginal iXY/=min(hX,hY)
    return iXY