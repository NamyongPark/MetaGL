import argparse
import pprint
import warnings

import dgl.base

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=dgl.base.DGLWarning)

import numpy as np
np.set_printoptions(precision=3, suppress=True)
from utils import setup_cuda, logger, set_seed, report_performance

import dataloader
from metagl import MetaGL


def main(args):
    perf_mat, meta_feat_mat, models, graphs, graph_domains = dataloader.load_data()

    M_splits, P_splits, train_index_vec, test_index_vec = \
        dataloader.load_train_test_splits(perf_mat, meta_feat_mat, graph_domains, args)

    num_graphs, num_models = perf_mat.shape
    num_meta_feats = meta_feat_mat.shape[1]

    metagl = MetaGL(
        num_models=num_models,
        metafeats_dim=num_meta_feats,
        epochs=args.epochs,
        device=args.device
    )
    logger.info(f"Running MetaGL...")
    set_seed(args.seed)

    """k-fold cross-validation"""
    for M_dict, P_dict in zip(M_splits, P_splits):
        metagl.train_predict(M_dict["train"],
                             M_dict["test"],
                             P_dict["train"],
                             P_dict["train_full"],
                             P_dict["test"])

    report_performance(metagl.name, metagl.eval_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set to -1 to use CPU.")
    parser.add_argument("--seed", type=int, default=1337,
                        help="random seed")
    parser.add_argument("--perf-nan-perc", type=float, default=0.0,
                        help="percentage of nans in the performance matrix")
    parser.add_argument("--k-fold-n-splits", type=int, default=5,
                        help="number of splits for k-fold cross validation")
    parser.add_argument("--epochs", type=int, default=500,
                        help="maximum number of training epochs")

    args = parser.parse_args()
    setup_cuda(args)
    print("\n[Settings]\n" + pprint.pformat(args.__dict__))
    set_seed(args.seed)

    main(args)
