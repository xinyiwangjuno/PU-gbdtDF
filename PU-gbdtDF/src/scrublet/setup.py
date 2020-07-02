from scrublet import Scrublet
from helper_functions import *
import scipy.io
import scipy.sparse
import pandas as pd
import numpy as np


def main(datafile, stat_filename, total_counts=None,
         sim_doublet_ratio=2.0, expected_doublet_rate=0.1,
         stdev_doublet_rate=0.02, random_state=0,
         max_iter=20, sample_rate=0.5, learn_rate=0.7,
         max_depth=1, split_points=100, p2u_pro=0.1, train_rate=0.7):
    print("Model parameters configuration:[data_file=%s,stat_file=%s,max_iter=%d,sample_rate=%f,learn_rate=%f,"
          "max_depth=%d,split_points=%d]" % (datafile, stat_filename, max_iter, sample_rate, learn_rate, max_depth, split_points))
    counts_matrix = scipy.sparse.load_npz(datafile)

    Scrublet(counts_matrix, stat_filename)


if __name__ == "__main__":
    input_filename = '/Users/junowang/Desktop/Final_CM/Codes/scrublet-modified/src/scrublet/GSM2560248_matrix.npz'
    stat_filename = 'output/scrublet-modified.csv'
    main(input_filename, stat_filename)
