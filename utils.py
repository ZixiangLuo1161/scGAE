import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, homogeneity_score, completeness_score
from clustering import *

def myscatter(Y, class_idxs, legend=False, ran=True, seed=229):
    if ran:
        np.random.seed(seed)
    Y = np.array(Y)
    fig, ax = plt.subplots(figsize=(6,4), dpi=300)
    classes = list(np.unique(class_idxs))
    markers = 'osD' * len(classes)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
    if ran:
        np.random.shuffle(colors)

    for i, cls in enumerate(classes):
        mark = markers[i]
        ax.plot(Y[class_idxs == cls, 0], Y[class_idxs == cls, 1], marker=mark,
                linestyle='', ms=4, label=str(cls), alpha=1, color=colors[i],
                markeredgecolor='black', markeredgewidth=0.15)
    if legend:
        ax.legend(bbox_to_anchor=(1.03, 1), loc=2, borderaxespad=0, fontsize=10, markerscale=2, frameon=False,
                  ncol=2, handletextpad=0.1, columnspacing=0.5)

    plt.xticks([])
    plt.yticks([])

    return ax


def dotsne(X, dim=2, ran=23):
    tsne = TSNE(n_components=dim, random_state=ran)
    Y_tsne = tsne.fit_transform(X)
    return Y_tsne


def dopca(X, dim=10):
    pcaten = PCA(n_components=dim)
    X_10 = pcaten.fit_transform(X)
    return X_10



def measure(true, pred):
    NMI = round(normalized_mutual_info_score(true, pred), 2)
    RAND = round(adjusted_rand_score(true, pred), 2)
    HOMO = round(homogeneity_score(true, pred), 2)
    COMP = round(completeness_score(true, pred), 2)
    return [NMI, RAND, HOMO, COMP]


def clustering(h, n_cluster, k=15, f="louvain"):
    from preprocessing import get_adj
    from clustering import louvain
    adj, adj_n = get_adj(h, k=k, pca=False)
    if f == "louvain":
        cl_model = louvain(level=0.5)
        cl_model.update(h, adj_mat=adj)
        labels = cl_model.labels
    elif f == "spectral":
        labels = SpectralClustering(n_clusters=n_cluster, affinity="precomputed", assign_labels="discretize",
                                    random_state=0).fit_predict(adj)
    elif f == "kmeans":
        labels = KMeans(n_clusters=n_cluster, random_state=0).fit(h).labels_
    return labels


def dpt(times, h):
    import scanpy as sc
    import scipy.stats as stats
    adata = sc.AnnData(X=h)
    adata.obs['times'] = times
    adata.uns['iroot'] = np.flatnonzero(adata.obs['times'] == 0)[0]
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=30, use_rep="X")
    sc.tl.diffmap(adata)
    sc.tl.dpt(adata)
    ds = adata.obs["dpt_pseudotime"]
    tau, p_value = stats.kendalltau(times, ds)
    return tau, ds

def get_centers_louvain(Y, adj):
    from clustering import louvain
    cl_model = louvain(level=0.5)
    cl_model.update(Y, adj_mat=model.adj)
    labels = cl_model.labels
    centers = computeCentroids(Y, labels)
    return centers, labels

def get_centers_spectral(Y, adj):
    from sklearn.cluster import SpectralClustering
    l = SpectralClustering(n_clusters=10,affinity="precomputed", assign_labels="discretize",random_state=0).fit_predict(adj)
    centers = computeCentroids(Y, l)
    return centers, l