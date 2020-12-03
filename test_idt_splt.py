import pandas as pd
def train_test_split(df, int_dt_window=100):
    df.reset_index(inplace=True)
    epochs = range(int_dt_window, max(df['date']) + 1, int_dt_window)
    for idt in epochs:
        print(f"train dt: 0 to {idt-1} and test dt: {idt} to {idt + int_dt_window}")

    # return [(df[(df['date'] < idt)].index.values,
    #          df[(df['date'] >= idt) &
    #             (df['date'] < idt + int_dt_window)
    #             ].index.values) for idt in epochs]


train = pd.read_csv('input/train.csv')
print(train)
train_test_split(train, 100)