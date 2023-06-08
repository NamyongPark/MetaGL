import os
from pathlib import Path

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold

from utils import logger


def load_data():
    root = Path(os.path.dirname(__file__)) / '..'
    data_root = root / 'data'
    data_fn = data_root / 'MetaGL_data.npz'
    npzfile = np.load(data_fn, allow_pickle=True)
    # noinspection PyUnresolvedReferences
    print('\nLoaded data variables:', npzfile.files)

    """Performance matrix"""
    P = np.matrix(npzfile['P'])  # MAP used as perf metric. shape=(301 graphs, 423 models)
    print(f"- num graphs: {P.shape[0]}, num models={P.shape[1]}\n")

    """Meta-graph features"""
    M = np.matrix(npzfile['M'])  # (301, 318). 318 features
    assert np.count_nonzero(np.isnan(M)) == 0

    """Other variables"""
    models = list(npzfile['models'])  # (423,)
    graphs = list(npzfile['graphs'])  # (301,)
    graph_domains = list(npzfile['y_graph_domain'])  # (301,)

    return P, M, models, graphs, graph_domains


def load_train_test_splits(perf_mat, meta_feat_mat, graph_domains, args):
    train_index_vec, test_index_vec = [], []
    kf = StratifiedKFold(n_splits=args.k_fold_n_splits, random_state=1, shuffle=True)
    for train_index, test_index in kf.split(perf_mat, graph_domains):
        train_index_vec.append(train_index)
        test_index_vec.append(test_index)
    assert len(train_index_vec) == len(test_index_vec), (len(train_index_vec), len(test_index_vec))

    M_norm = sklearn.preprocessing.minmax_scale(meta_feat_mat.copy(), axis=0)  # scale each col (meta-feature)
    M_splits = []
    for train_index, test_index in zip(train_index_vec, test_index_vec):
        M_train, M_test = M_norm[train_index], M_norm[test_index]  # meta-feature matrix
        M_splits.append({"train": M_train, "test": M_test, "all": M_norm})

    P_tmp = perf_mat.copy()
    P_splits = []
    for split_i, (train_index, test_index) in enumerate(zip(train_index_vec, test_index_vec)):
        P_train_full, P_test = P_tmp[train_index], P_tmp[test_index]

        if args.perf_nan_perc > 0:
            P_train = create_partially_observed_perf_matrix(P_train_full, args.perf_nan_perc)
        else:
            P_train = P_train_full

        P_splits.append({"train": P_train, "train_full": P_train_full, "test": P_test, "all": P_tmp})

    assert len(M_splits) == len(P_splits)
    return M_splits, P_splits, train_index_vec, test_index_vec


def create_partially_observed_perf_matrix(P, perc_to_replace_with_nan=0.5):
    """
    Create a partially observed performance matrix
    by replacing performance entries of P with NaNs uniformly at random
    """
    assert 0 <= perc_to_replace_with_nan <= 1, perc_to_replace_with_nan
    P_partially_observed = np.array(P.copy())

    num_total = P_partially_observed.size
    num_replace = int(P_partially_observed.size * perc_to_replace_with_nan)
    P_partially_observed.ravel()[np.random.choice(num_total, num_replace, replace=False)] = np.nan
    logger.info(f"Updated P: {np.count_nonzero(np.isnan(P_partially_observed))} NaNs "
                f"out of {P_partially_observed.size} elements in P - perc_to_replace_with_nan={perc_to_replace_with_nan}")

    return P_partially_observed


if __name__ == '__main__':
    load_data()
