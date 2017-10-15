from __future__ import print_function

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas as pd

from mpl_toolkits.axes_grid1 import Grid


def bar_line_plot(ax1, time, data1, data2, c1, c2,
                  xmin, ymin1, ymin2,
                  xmax, ymax1, ymax2,
                  show_xlabel,
                  show_ylabel_1,
                  show_ylabel_2,
                  annotate_col,
                  col_annotation,
                  annotate_row,
                  row_annotation,):
    """

    Parameters
    ----------
    ax : axis
        Axis to put two scales on

    time : array-like
        x-axis values for both datasets

    data1: array-like
        Data for left hand scale

    data2 : array-like
        Data for right hand scale

    c1 : color
        Color for line 1

    c2 : color
        Color for line 2

    Returns
    -------
    ax : axis
        Original axis
    ax2 : axis
        New twin axis
    """
    ax2 = ax1.twinx()

    ax1.bar(time, data2, color=c2, width=10)

    if show_xlabel:
        ax1.set_xlabel('Training Step')
    else:
        ax1.xaxis.set_ticklabels([])

    if show_ylabel_1:
        ax1.set_ylabel('Queue Size')
    else:
        ax1.yaxis.set_ticklabels([])

    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin2, ymax2)

    ax2.plot(time, data1, color=c1, alpha=0.75)

    if show_ylabel_2:
        ax2.set_ylabel('Mean Single Batch \n Inference Running Time')
    else:
        ax2.yaxis.set_ticklabels([])

    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin1, ymax1)

    if annotate_col:
        pad = 10
        ax1.annotate(col_annotation, xy=(0.5, 1), xytext=(0, pad),
                     xycoords='axes fraction', textcoords='offset points',
                     size='large', ha='center', va='baseline')

    if annotate_row:
        pad = -70
        ax1.annotate(row_annotation, xy=(0, 0.75), xytext=(pad, 0),
                     rotation=90,
                     xycoords='axes fraction', textcoords='offset points',
                     size='large', ha='center', va='baseline')

    return ax1, ax2


plt.style.use('ggplot')

df = pd.read_csv('C:/Users/Justi/Research/logs/queue_exhaustion_out.csv')

max_running_time = np.max(df.running_time)
print(max_running_time)
min_running_time = 0

max_queue_size = np.max(df.queue_size)
print(max_queue_size)
min_queue_size = 0

max_step_num = np.max(df.step_num)
print(max_queue_size)
min_step_num = np.min(df.step_num)

subplot_x_str = str(len(df.thread_count.unique()))
subplot_y_str = str(len(df.batch_size.unique()))


fig = plt.figure()

plot_num = 0


for i, tc in enumerate(df.thread_count.unique()):

    # print(df.loc[df['thread_count'] == tc])

    for j, bs in enumerate(df.batch_size.unique()):

        plot_num += 1

        run_df = df.loc[(df['batch_size'] == bs) & (df['thread_count'] == tc)]

        # print(run_df)
        # Create some mock data
        t = run_df['step_num']
        s1 = run_df['running_time']
        s2 = run_df['queue_size']

        show_xlabel = len(df.thread_count.unique()) == (i + 1)
        show_label_1 = j == 0
        show_label_2 = len(df.batch_size.unique()) == (j + 1)

        annotate_col = i == 0
        col_annotation = 'Batch Size = %d' % bs

        annotate_row = j == 0
        row_annotation = 'Thread Count = %d' % tc

        # Create axes
        
        ax = fig.add_subplot(len(df.thread_count.unique()),
                             len(df.batch_size.unique()),
                             plot_num)
        ax1, ax2 = bar_line_plot(ax, t, s1, s2, 'r', 'b',
                                 min_step_num,
                                 min_running_time,
                                 min_queue_size,
                                 max_step_num,
                                 max_running_time,
                                 max_queue_size,
                                 show_xlabel,
                                 show_label_1,
                                 show_label_2,
                                 annotate_col,
                                 col_annotation,
                                 annotate_row,
                                 row_annotation)

plt.suptitle("TensorFlow Queue Exhaustion on Hokule'a")

plt.show()
