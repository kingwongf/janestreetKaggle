# coding=utf-8
import pandas as pd
import os


# number of rows found: 2390490 (more than 2 millions)
# number of days : 500
# number of records per day: 4781 ca.


def read_big_csv(filepath: str, s: int = None, n: int = None) -> pd.DataFrame:
    if n is not None:
        rows = pd.read_csv(filepath, header=0, nrows=n)
    else:
        rows = pd.read_csv(filepath, header=0, skiprows=s)
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


file = '../jane-street-market-prediction/train.csv'
os.chdir(r'C:\Kaggle-King\janestreetKaggle')

if __name__ == '__main__':
    r = count_rows(filepath=file)
    view_top = read_big_csv(filepath=file, n=100)
    # view_bottom = read_big_csv(filepath=file, s=2390400)

