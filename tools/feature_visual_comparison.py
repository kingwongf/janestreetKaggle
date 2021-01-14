# coding=utf-8
import numpy as np
import janestreetKaggle.tools.big_csv_reader as bcr
from matplotlib import pyplot as plt

# THIS FILE EXTRACT A SMALL SUBSET OF THE DATA TO EXPLORE THE CORRELATION AMONG DIFFERENT FEATURES.
# USE "PLOT_FEATURE" TO PLOT ONE FEATURE
# USE "VIA" TO REMOVE ONE PLOT


def trend_with_exp_weighted_av(valori, n_giorni):
    b = 1 - 1/n_giorni
    lv = len(valori)
    vt = np.zeros(lv)
    # vt[0] = (1-b)*valori[0]
    for t in range(1, lv):
        vt[t] = b*vt[t-1] + (1-b)*valori[t]
    vt[1:] = vt[1:] * 1/(1-b**np.arange(1, lv))
    shift = round(n_giorni/2)
    vt_adjusted = np.roll(vt, -shift)
    vt_adjusted[-shift:] = 0
    return vt_adjusted, shift


def load_data():
    data = bcr.read_big_csv(filepath=bcr.file, n=20000)
    data_sub = data[(data['ts_id'] >= 12287) & (data['ts_id'] <= 12406)]
    return data_sub


def plot_feature(data, n):
    feature_name = 'feature_' + str(n)
    values = data[feature_name]
    # f = ip.interp1d(values.index, values.values, kind='cubic')

    # values.plot()
    # plt.plot(values.index, f(values.index))
    average, sh = trend_with_exp_weighted_av(valori=values.values, n_giorni=10)
    my_plot = plt.plot(values.index[sh:], average[:-sh], label=str(n))
    plt.legend()
    return my_plot


def remove_feature(p):
    l0 = p.pop(0)
    l0.remove()


def compare_features(data):
    plt.figure()
    for f in range(20):
        plot_feature(data=data, n=f+1)
    plt.show()


d = load_data()
compare_features(data=d)
