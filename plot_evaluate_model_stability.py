from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd


def dual_scatter(ax1, time, data1, data2, c1, c2,
                 xmin, ymin1, ymin2,
                 xmax, ymax1, ymax2,
                 show_xlabel,
                 show_ylabel_1,
                 show_ylabel_2,
                 annotate_col,
                 col_annotation,
                 annotate_row,
                 row_annotation,):

    ax2 = ax1.twinx()

    ax1.plot(time, data1, color=c1, alpha=0.2)
    ax2.plot(time, data2, color=c2, alpha=0.2)

    ax1.scatter(time, data1, color=c1, alpha=0.8, marker='^')
    ax2.scatter(time, data2, color=c2, alpha=0.8)

    # ax1.set_xlim(xmin, xmax)
    # ax2.set_xlim(xmin, xmax)

    ax2.set_ylim(ymin2, ymax2)
    ax1.set_ylim(ymin1, ymax1)

    ax1.set_xscale("log", nonposx='clip')
    ax2.set_xscale("log", nonposx='clip')

    ax1.set_yscale("log", nonposy='clip')
    ax2.set_yscale("log", nonposy='clip')

    if show_xlabel:
        ax1.set_xlabel('Batch Interval')
    else:
        ax1.xaxis.set_ticklabels([])

    if show_ylabel_2:
        ax2.set_ylabel('Min Achieved \nValidation Loss')
    else:
        ax2.yaxis.set_ticklabels([])

    if show_ylabel_1:
        ax1.set_ylabel('Mean Single \nBatch Inference \nRunning Time')
    else:
        ax1.yaxis.set_ticklabels([])

    if annotate_col:
        pad = 10
        ax1.annotate(col_annotation, xy=(0.5, 1), xytext=(0, pad),
                     xycoords='axes fraction', textcoords='offset points',
                     size='large', ha='center', va='baseline')

    if annotate_row:
        pad = -100
        ax1.annotate(row_annotation, xy=(0, 0.55), xytext=(pad, 0),
                     rotation=90,
                     xycoords='axes fraction', textcoords='offset points',
                     size='large', ha='center', va='baseline')

    return ax1, ax2


plt.style.use('ggplot')

df = pd.read_csv('C:/Users/Justi/Research/log/evaluate_model_stability/evaluate_model_stability.csv')

max_batch_interval = np.max(df.batch_interval)
min_batch_interval = np.min(df.batch_interval)

subplot_x_str = str(len(df.thread_count.unique()))
subplot_y_str = str(len(df.batch_size.unique()))


fig = plt.figure()

plot_num = 0

# matshow: x is batch interval, y is batch size, c is val_loss
# matshow: x is batch interval, y is batch sizen_time
# matshow: x is batch interval, y is batch size, c is re, c is meal_val_loss * rel_mean_time

# A given HPC node is a line trading along num_workers and num_threads

# Choose number of workers, gross batch size, and batch_int to drive the optimal c, 

for i, tc in enumerate(df.thread_count.unique()):

    # print(df.loc[df['thread_count'] == tc])

    for j, bs in enumerate(df.batch_size.unique()):

        plot_num += 1

        t = df.batch_interval.unique()

        s1 = []

        s2 = []

        for k, bi in enumerate(df.batch_interval.unique()):

            run_df = df.loc[(df['batch_size'] == bs) &
                            (df['thread_count'] == tc) &
                            (df['batch_interval'] == bi)]

            # We'll need a mean here...

            # # Get the mean val_loss over reps, binned by batch interval.
            # s1 = np.min(run_df['val_loss'])

            # # Get the mean mean_running_time over reps, binned by batch interval.
            # s2 = np.mean(run_df['mean_running_time'])

            s1.append(np.min(run_df['val_loss']))

            s2.append(np.mean(run_df['mean_running_time']))

        show_xlabel = len(df.thread_count.unique()) == (i + 1)
        show_label_1 = j == 0
        show_label_2 = len(df.batch_size.unique()) == (j + 1)

        annotate_col = i == 0
        col_annotation = 'Batch Size = %d' % bs

        annotate_row = j == 0
        row_annotation = 'Thread \n Count = %d' % tc

        # min_s2 = 0
        # max_s2 = 0.04

        # min_s1 = 0
        # max_s1 = 0.2

        min_s2 = 0.001
        max_s2 = 1

        min_s1 = 0.001
        max_s1 = 1

        # Create axes.
        ax = fig.add_subplot(len(df.thread_count.unique()),
                             len(df.batch_size.unique()),
                             plot_num)

        ax1, ax2 = dual_scatter(ax, t, s1, s2, 'r', 'b',
                                min_batch_interval,
                                min_s1,
                                min_s2,
                                max_batch_interval,
                                max_s1,
                                max_s2,
                                show_xlabel,
                                show_label_1,
                                show_label_2,
                                annotate_col,
                                col_annotation,
                                annotate_row,
                                row_annotation)

        s1_label = mlines.Line2D([], [], color='r', marker='^',
                                 markersize=7,
                                 label='Min Achieved \nValidation Loss')

        s2_label = mlines.Line2D([], [], color='b', marker='o',
                                 markersize=7,
                                 label='Mean Single Batch \nInference Running Time')

        plt.legend(handles=[s1_label, s2_label], loc=4)

plt.suptitle("Validation Loss and Mean Optimization Running Time by Batch Interval")

plt.show()
