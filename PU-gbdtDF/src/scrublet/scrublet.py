from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.pyplot as plt
from data import DataSet
from model import Model
from mytree import *
import scipy.io
import scipy.sparse
import numpy as np
import time
from sklearn.metrics import roc_auc_score
from helper_functions import print_optional, pipeline_pca, pipeline_truncated_svd, pipeline_normalize_variance, \
    pipeline_mean_center, pipeline_zscore, pipeline_log_transform, pipeline_normalize, pipeline_apply_gene_filter, \
    pipeline_get_gene_filter


class Scrublet():

    def __init__(self, counts_matrix, stat_filename, total_counts=None,
                 sim_doublet_ratio=2.0, expected_doublet_rate=0.1,
                 stdev_doublet_rate=0.02, random_state=0,
                max_iter = 20, sample_rate = 0.5, learn_rate = 0.7,
                 max_depth = 1, split_points = 1000, p2u_pro = 0.1, train_rate = 0.7):

        if not scipy.sparse.issparse(counts_matrix):
            counts_matrix = scipy.sparse.csc_matrix(counts_matrix)
        elif not scipy.sparse.isspmatrix_csc(counts_matrix):
            counts_matrix = counts_matrix.tocsc()

        self.counts_matrix_d = DataSet(counts_matrix)
        self.counts_matrix_d.describe()

        # initialize counts matrices
        self._E_obs = counts_matrix
        self._E_sim = None
        self._E_obs_norm = None
        self._E_sim_norm = None
        self.max_iter = max_iter
        self.sample_rate = sample_rate
        self.learn_rate = learn_rate
        self.max_depth = max_depth
        self.split_points = split_points
        self.p2u_pro = p2u_pro
        self.train_rate = train_rate
        self.stat_filename = stat_filename

        if total_counts is None:
            self._total_counts_obs = self._E_obs.sum(1).A.squeeze()
        else:
            self._total_counts_obs = total_counts

        self._gene_filter = np.arange(self._E_obs.shape[1])
        self._embeddings = {}

        self.sim_doublet_ratio = sim_doublet_ratio
        self.expected_doublet_rate = expected_doublet_rate
        self.stdev_doublet_rate = stdev_doublet_rate
        self.random_state = random_state



    ######## Core Scrublet functions ########

    def scrub_doublets(self, synthetic_doublet_umi_subsampling=1.0,
                       min_counts=3, min_cells=3, min_gene_variability_pctl=85,
                       log_transform=False, mean_center=True, normalize_variance=True, n_prin_comps=30, verbose=True):
        t0 = time.time()

        self._E_sim = None
        self._E_obs_norm = None
        self._E_sim_norm = None
        self._gene_filter = np.arange(self._E_obs.shape[1])

        print_optional('Preprocessing...', verbose)
        pipeline_normalize(self)
        pipeline_get_gene_filter(self, min_counts=min_counts, min_cells=min_cells,
                                 min_gene_variability_pctl=min_gene_variability_pctl)
        pipeline_apply_gene_filter(self)

        print_optional('Simulating doublets...', verbose)
        self.simulate_doublets(sim_doublet_ratio=self.sim_doublet_ratio,
                               synthetic_doublet_umi_subsampling=synthetic_doublet_umi_subsampling)
        pipeline_normalize(self, postnorm_total=1e6)
        if log_transform:
            pipeline_log_transform(self)
        if mean_center and normalize_variance:
            pipeline_zscore(self)
        elif mean_center:
            pipeline_mean_center(self)
        elif normalize_variance:
            pipeline_normalize_variance(self)

        if mean_center:
            print_optional('Embedding transcriptomes using PCA...', verbose)
            pipeline_pca(self, n_prin_comps=n_prin_comps, random_state=self.random_state)
        else:
            print_optional('Embedding transcriptomes using Truncated SVD...', verbose)
            pipeline_truncated_svd(self, n_prin_comps=n_prin_comps, random_state=self.random_state)


        t1 = time.time()
        print_optional('Elapsed time: {:.1f} seconds'.format(t1 - t0), verbose)


    def simulate_doublets(self, sim_doublet_ratio=None, synthetic_doublet_umi_subsampling=1.0):
        ''' Simulate doublets by adding the counts of random observed transcriptome pairs.

        Arguments
        ---------
        sim_doublet_ratio : float, optional (default: None)
            Number of doublets to simulate relative to the number of observed 
            transcriptomes. If `None`, self.sim_doublet_ratio is used.

        synthetic_doublet_umi_subsampling : float, optional (defuault: 1.0) 
            Rate for sampling UMIs when creating synthetic doublets. If 1.0, 
            each doublet is created by simply adding the UMIs from two randomly 
            sampled observed transcriptomes. For values less than 1, the 
            UMI counts are added and then randomly sampled at the specified
            rate.

        Sets
        ----
        doublet_parents_
        '''

        if sim_doublet_ratio is None:
            sim_doublet_ratio = self.sim_doublet_ratio
        else:
            self.sim_doublet_ratio = sim_doublet_ratio

        self.n_obs = self._E_obs.shape[0]
        self.n_sim = int(self.n_obs * sim_doublet_ratio)

        np.random.seed(self.random_state)
        pair_ix = np.random.randint(0, self.n_obs, size=(self.n_sim, 2))

        E1 = self._E_obs[pair_ix[:,0],:]
        E2 = self._E_obs[pair_ix[:,1],:]
        tots1 = self._total_counts_obs[pair_ix[:,0]]
        tots2 = self._total_counts_obs[pair_ix[:,1]]
        if synthetic_doublet_umi_subsampling < 1:
            self._E_sim, self._total_counts_sim = subsample_counts(E1+E2, synthetic_doublet_umi_subsampling, tots1+tots2, random_seed=self.random_state)
        else:
            self._E_sim = E1+E2
            self._total_counts_sim = tots1+tots2
        self.doublet_parents_ = pair_ix
        return


    def classifier(self, exp_doub_rate=0.1, stdev_doub_rate=0.03):

        stat_file = open(self.stat_filename, "w+", encoding='gbk')

        stat_file.write("iteration\taverage_loss_in_train_data\tprediction_accuracy_on_test_data\taverage_loss_in_test "
                        "data\n")
        doub_labels = np.concatenate((np.zeros(self.n_obs, dtype=int),
                                      np.ones(self.n_sim, dtype=int)))

        model = Model(self.max_iter, self.sample_rate, self.learn_rate, self.max_depth, self.split_points)
        train_data = self.counts_matrix_d.train_data_id(self.p2u_pro, self.train_rate)
        test_data = self.counts_matrix_d.test_data_id(self.p2u_pro, self.train_rate)
        model.train(self.counts_matrix_d, train_data, stat_file, test_data)
        test_data_predict, x, y = model.test(self.counts_matrix_d, test_data)
        y_true = []
        for id in test_data:
            y_true.append(self.counts_matrix_d.get_instance(id)['label'])
        y_pred = test_data_predict
        y_pred = [int(id) for id in y_pred]
        print(y_pred)
        auc_score = roc_auc_score(y_true, y_pred)
        print('auc_score=', auc_score)
        stat_file.close()
        



    


