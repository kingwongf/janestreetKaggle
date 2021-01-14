# coding=utf-8
import pandas as pd
import os
import time


file = '../jane-street-market-prediction/train.csv'
file_lenght = 2390492
os.chdir(r'C:\Kaggle-King\janestreetKaggle')

# number of rows found: 2390491 (more than 2 millions)
# number of days : 500
# number of records per day: 4781 ca.


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
        rows_to_skip = list(range(1, s+1)) + list(range(s+1+n, file_lenght))
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
    with open(filepath, 'r') as f0:
        f0.seek(0, 2)  # Seek @ EOF
        fsize = f0.tell()  # Get Size
        f0.seek(max(fsize - 4096, 0), 0)  # Set pos @ last n chars
        lines = f0.readlines()  # Read to end
        print(lines)
        # print('number of rows:', row_count)
    return lines


if __name__ == '__main__':
    option_1 = False

    if option_1:
        data = read_big_csv(s=50, n=20)
