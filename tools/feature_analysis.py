# coding=utf-8
from big_csv_reader import *


def make_stats(n: int = None) -> (pd.DataFrame, int):
    stats = {}
    my_data = read_big_csv(n=n)
    stats['minima'] = my_data.min()
    stats['maxima'] = my_data.max()
    stats['average'] = my_data.mean()
    stats['variance'] = my_data.var()
    number_of_records = len(my_data)
    stats_df = pd.DataFrame(data=stats)
    stats_df.drop(['date', 'ts_id'])
    return stats_df, number_of_records


def find_correlations(dataset: pd.DataFrame):
    # can this be improved comparing only the columns where we have "non-Nan" values?
    threshold = 0.85
    dataset_new = dataset.drop(columns=['date', 'weight', 'feature_0', 'resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4', 'ts_id'])
    t0 = time.time()
    print('computing the correlation matrix ... ')
    c = dataset_new.corr()
    t1 = time.time()
    print('computed in %f seconds.\nSaving ...' % (t1-t0))
    c.to_csv(r'C:\Kaggle-King\janestreetKaggle\tools\feature_correlations_coeff_t%s.csv' % str(threshold))
    print('DONE.\nDropping correlated columns ... ')
    for i in range(1, 130):
        feature = 'feature_' + str(i)
        if feature in c.columns:
            condition = feature + '>=' + str(threshold)
            correlated = set(c.query(condition).index.to_list()) - {feature}
            if bool(correlated):
                c.drop(columns=list(correlated), inplace=True)
                c.drop(correlated, inplace=True)
    return list(c.columns)

    # take only the non-Nan common values


def plot_feature_stats():
    stats = pd.read_csv(r'C:\Kaggle-King\janestreetKaggle\tools\feature_stats.csv', index_col=0)
    stats.plot()


def make_stats_on_weighted_resp(n: int = None):
    stats = {}
    my_data = read_big_csv(n=n)
    my_data = my_data[['resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4']].multiply(my_data['weight'], axis='index')
    stats['minima'] = my_data.min()
    stats['maxima'] = my_data.max()
    stats['average'] = my_data.mean()
    stats['variance'] = my_data.var()
    number_of_records = len(my_data)
    stats_df = pd.DataFrame(data=stats)
    return stats_df, number_of_records


if __name__ == '__main__':
    option_1 = False
    option_2 = False

    if option_1:
        r = find_correlations(dataset=read_big_csv())
        print(''.join(['non-correlated-features:\n', r, '(N: ', len(r), ')']))
        f = open(r'C:\Kaggle-King\janestreetKaggle\tools\features_not_correlated.txt', 'wt')
        f.write('\n'.join(r))
        f.close()

    if option_2:
        df, nr = make_stats_on_weighted_resp()
        df.to_csv(r'C:\Kaggle-King\janestreetKaggle\tools\weighted_responses_stats.csv')