import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors


def compute_knn_scaled_scores(X_test, X_ref, K=5):
    def l2_normalize(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    if X_test.shape[0] == 0:
        return np.empty((0, 1))

    X_test = l2_normalize(X_test)
    X_ref = l2_normalize(X_ref)

    dists = cosine_distances(X_test, X_ref)  # shape: (N_test, N_ref)

    # k-NN计算
    nbrs = NearestNeighbors(n_neighbors=K + 1, metric='cosine').fit(X_ref)
    _, indices = nbrs.kneighbors(X_ref)
    neighbor_indices = indices[:, 1:]

    # 计算每个参考样本的密度
    ref_density = []
    for i in range(X_ref.shape[0]):
        y = X_ref[i]
        y_neighbors = X_ref[neighbor_indices[i]]
        local_dist = cosine_distances(y.reshape(1, -1), y_neighbors).mean()
        ref_density.append(local_dist)

    ref_density = np.array(ref_density)

    # k-NN重缩放分数的计算
    scaled_scores = []
    for i in range(X_test.shape[0]):
        score = dists[i] / (ref_density + 1e-8)
        scaled_scores.append(0.5 * score.min())

    scaled_scores = np.array(scaled_scores)
    return scaled_scores


def compute_gwrp_scaled_scores(X_test, X_ref, r=0.5, K=20):
    def l2_normalize(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    if X_test.shape[0] == 0:
        return np.empty((0, 1))

    # 步骤1：L2归一化
    X_test = l2_normalize(X_test)
    X_ref = l2_normalize(X_ref)

    # 步骤2：计算每个参考样本的局部加权密度因子（只和X_ref有关）
    nbrs = NearestNeighbors(n_neighbors=K + 1, metric='cosine').fit(X_ref)
    distances, indices = nbrs.kneighbors(X_ref)
    neighbor_indices = indices[:, 1:]  # 去除自己

    weighted_densities = []
    for i in range(X_ref.shape[0]):
        y = X_ref[i]
        y_neighbors = X_ref[neighbor_indices[i]]

        local_dists = cosine_distances(y.reshape(1, -1), y_neighbors).flatten()  # shape = (K,)
        weights = r ** np.arange(K)  # r^0, r^1, ..., r^{K-1}
        weighted_density = np.sum(local_dists * weights)
        weighted_densities.append(weighted_density + 1e-8)  # 避免除零

    weighted_densities = np.array(weighted_densities)  # shape = (N_ref,)

    # 步骤3：对每个测试样本，计算到所有参考样本的距离，并除以对应的加权密度
    dists = cosine_distances(X_test, X_ref)  # shape = (N_test, N_ref)

    gwrp_scores = []
    for i in range(X_test.shape[0]):
        normalized_dists = dists[i] / weighted_densities  # 对每个参考样本的距离除以密度
        score = 0.5 * np.min(normalized_dists)  # 取最小归一化距离 × 0.5
        gwrp_scores.append(score)

    return np.array(gwrp_scores)
