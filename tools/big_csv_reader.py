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
    if s is not None or n is not None:
        rows_to_skip = list(range(0, s)) + list(range(s+n, file_lenght))
        rows = pd.read_csv(filepath, header=0, skiprows=rows_to_skip)
    else:
        rows = pd.read_csv(filepath, header=0)
    t1 = time.time()
    print('loaded in %f seconds' % (t1-t0))
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


def plot_feature_stats():
    stats = pd.read_csv(r'C:\Kaggle-King\janestreetKaggle\tools\feature_stats.csv', index_col=0)
    stats.plot()


os.chdir(r'C:\Kaggle-King\janestreetKaggle')


if __name__ == '__main__':
    # r = count_rows(filepath=file)
    # view_top = read_big_csv(filepath=file, n=100)
    # view_bottom = read_big_csv(filepath=file, s=2390400)
    dataf, nr = make_stats()
    dataf.to_csv(r'C:\Kaggle-King\janestreetKaggle\tools\feature_stats.csv')
    print('n records = ', nr)
