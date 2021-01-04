# coding=utf-8
import pandas as pd
import os
import time


# number of rows found: 2390491 (more than 2 millions)
# number of days : 500
# number of records per day: 4781 ca.

file = '../jane-street-market-prediction/train.csv'
file_lenght = 2390490


def assess_time_to_load(filepath: str = file, n: int = 0):
    t0 = time.time()
    print('reading %d records' % n)
    read_big_csv(filepath=filepath, n=n)
    t1 = time.time()
    print('read in %f seconds' % (t1 - t0))


def read_big_csv(filepath: str = file, s: int = None, n: int = None) -> pd.DataFrame:
    print('loading records')
    t0 = time.time()
    if s is not None and n is not None:
        rows_to_skip = list(range(0, s)) + list(range(s+n, file_lenght))
        rows = pd.read_csv(filepath, header=0, skiprows=rows_to_skip)
    elif n is not None:
        rows = pd.read_csv(filepath, header=0, nrows=n)
    else:
        rows = pd.read_csv(filepath, header=0)
    t1 = time.time()
    print('loaded in %f seconds' % (t1-t0))
    print('%d records read' % len(rows))
    return rows


def count_rows(filepath: str):
    with open(filepath, 'r') as f:
        f.seek(0, 2)  # Seek @ EOF
        fsize = f.tell()  # Get Size
        f.seek(max(fsize - 4096, 0), 0)  # Set pos @ last n chars
        lines = f.readlines()  # Read to end
        print(lines)
        # print('number of rows:', row_count)
    return lines


def make_stats(n: int = None) -> (pd.DataFrame, int):
    stats = {}
    data = read_big_csv(n=n)
    stats['minima'] = data.min()
    stats['maxima'] = data.max()
    stats['average'] = data.mean()
    stats['variance'] = data.var()
    number_of_records = len(data)
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


os.chdir(r'C:\Kaggle-King\janestreetKaggle')


if __name__ == '__main__':
    option_1 = False
    option_2 = True
    # r = count_rows(filepath=file)
    # view_top = read_big_csv(filepath=file, n=100)
    # view_bottom = read_big_csv(filepath=file, s=2390400)
    # dataf, nr = make_stats()
    # dataf.to_csv(r'C:\Kaggle-King\janestreetKaggle\tools\feature_stats.csv')
    # print('n records = ', nr)
    if option_1:
        r = find_correlations(dataset=read_big_csv())
        print(''.join(['non-correlated-features:\n', r, '(N: ', len(r), ')']))
        f = open(r'C:\Kaggle-King\janestreetKaggle\tools\features_not_correlated.txt', 'wt')
        f.write('\n'.join(r))
        f.close()

    if option_2:
        data = read_big_csv(s=50, n=20)
