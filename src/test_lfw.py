#coding=utf-8
import sys
import numpy as np
from common import *
from scipy.io import loadmat
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from joint_bayesian import *



def excute_train(train_data="../data/lbp_WDRef.mat", train_label="../data/id_WDRef.mat", result_fold="../result/"):
    data  = loadmat(train_data)['lbp_WDRef']
    label = loadmat(train_label)['id_WDRef']
    print data.shape, label.shape

    # data predeal
    data = data_pre(data)

    # pca training.
    pca = PCA_Train(data, result_fold)
    data_pca = pca.transform(data)

    data_to_pkl(data_pca, result_fold+"pca_wdref.pkl")
    JointBayesian_Train(data_pca, label, result_fold)


def excute_test(pairlist="../data/pairlist_lfw.mat", test_data="../data/lbp_lfw.mat", result_fold="../result/"):
    with open(result_fold+"A.pkl", "rb") as f:
        A = pickle.load(f)
    with open(result_fold+"G.pkl", "rb") as f:
        G = pickle.load(f)

    pair_list = loadmat(pairlist)['pairlist_lfw']
    test_Intra = pair_list['IntraPersonPair'][0][0] - 1
    test_Extra = pair_list['ExtraPersonPair'][0][0] - 1


    print test_Intra, test_Intra.shape
    print test_Extra, test_Extra.shape

    data  = loadmat(test_data)['lbp_lfw']
    data  = data_pre(data)

    clt_pca = joblib.load(result_fold+"pca_model.m")
    data = clt_pca.transform(data)
    data_to_pkl(data, result_fold+"pca_lfw.pkl")

    data = read_pkl(result_fold+"pca_lfw.pkl")
    print data.shape

    dist_Intra = get_ratios(A, G, test_Intra, data)
    dist_Extra = get_ratios(A, G, test_Extra, data)

    dist_all = dist_Intra + dist_Extra
    dist_all = np.asarray(dist_all)
    label    = np.append(np.repeat(1, len(dist_Intra)), np.repeat(0, len(dist_Extra)))

    data_to_pkl({"distance": dist_all, "label": label}, result_fold+"result.pkl")


if __name__ == "__main__":
    excute_train()
    excute_test()
    excute_performance("../result/result.pkl", -16.9, -16.6, 0.01)
