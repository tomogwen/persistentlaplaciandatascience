
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import sklearn


def plot_one_lap(df, idx):
    ex = df.iloc[idx]
    start, end = ex['start_val'], ex['end_val']
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    sns.heatmap(ex['radial_filtration'][0], ax=axes[0], xticklabels=False, yticklabels=False)
    sns.heatmap(ex['radial_filtration'][0] < start, ax=axes[1], xticklabels=False, yticklabels=False, cbar=False)
    sns.heatmap(ex['radial_filtration'][0] < end, ax=axes[2], xticklabels=False, yticklabels=False, cbar=False)

    sns.lineplot(np.sort(ex['eigs'][0]), ax=axes[3], label='L0')
    sns.lineplot(np.sort(ex['eigs'][1]), ax=axes[3], label='L1')
    axes[3].xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))

    axes[0].set_title('Radial filtration')
    axes[1].set_title('Filtration at ' + str(start))
    axes[2].set_title('Filtration at ' + str(end))
    axes[3].set_title('PersLap eigenvalues ' + str(start) + '↪' + str(end))

    plt.plot()


def all_filt_plotting(df, id_):
    min_ = 5
    max_ = 30
    inc = 5

    data = df[df['id'] == id_]
    # func = np.min

    num_start_vals = len(np.unique(data['start_val']))
    num_end_vals = len(np.unique(data['end_val']))

    min_vals = np.zeros((2, num_start_vals, num_end_vals))
    mean_vals = np.zeros((2, num_start_vals, num_end_vals))

    for i, k in enumerate(range(min_, max_ + inc, inc)):
        for j, l in enumerate(range(k, max_ + inc, inc)):
            dat = data[(data['start_val'] == k) & (data['end_val'] == l)]
            for q, vals_q in enumerate(dat['eigs'].values[0]):
                min_vals[q, i, j] = np.min(vals_q)
                mean_vals[q, i, j] = np.mean(vals_q)

    vmin_mean = 0  # np.min(mean_vals)
    vmax_mean = 3.5  # np.max(mean_vals)

    vmin_min = 0  # np.min(min_vals)
    vmax_min = 0.5  # np.max(min_vals)

    start, end = 10, 30
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    sns.heatmap(data['radial_filtration'].iloc[0][0], ax=axes[0], xticklabels=False, yticklabels=False, cbar=False)
    sns.heatmap(min_vals[0], ax=axes[1], cbar=False, vmin=vmin_min, vmax=vmax_min)
    sns.heatmap(min_vals[1], ax=axes[2], cbar=True, vmin=vmin_min, vmax=vmax_min)
    sns.heatmap(mean_vals[0], ax=axes[3], cbar=False, vmin=vmin_mean, vmax=vmax_mean)
    sns.heatmap(mean_vals[1], ax=axes[4], cbar=True, vmin=vmin_mean, vmax=vmax_mean)

    axes[0].set_title('Radial filtration')
    axes[1].set_title('L0 min eigs')
    axes[2].set_title('L1 min eigs')
    axes[3].set_title('L0 mean eigs')
    axes[4].set_title('L1 mean eigs')

    for i in range(1, 5):
        axes[i].set_xticklabels([5, 10, 15, 20, 25, 30])
    axes[1].set_yticklabels(['+0', '+5', '+10', '+15', '+20', '+25'])
    for i in range(2, 5):
        axes[i].set_yticklabels([])
        axes[i].set_yticks([])

    plt.plot()


if __name__ == '__main__':
    df = pd.read_pickle('results2_mnist.pkl')

    plot_one_lap(df, 13)
    plot_one_lap(df, 3330)

    all_filt_plotting(df, 0)
    all_filt_plotting(df, 23)
