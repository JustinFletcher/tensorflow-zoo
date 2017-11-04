import os
import sys
import time
import argparse
import functools
import itertools
import tensorflow as tf

sys.path.append("/.")
from baseline_model import *

if __name__ == '__main__':

    thread_counts = [16, 32, 64]
    batch_sizes = [8, 16, 32, 64]
    batch_intervals = [1, 2, 3, 4]

    for (tc, bs, bi) in itertools.product(thread_counts,
                                          batch_sizes,
                                          batch_intervals):
        print((tc, bs, bi))
