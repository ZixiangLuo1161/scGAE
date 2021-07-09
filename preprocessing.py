import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.neighbors import kneighbors_graph
from utils import dopca
import scanpy as sc
from anndata import AnnData

def get_adj(count, k=160, pca=30, mode="connectivity"):
    if pca:
        countp = dopca(count, dim=pca)
    else:
        countp = count
    A = kneighbors_graph(countp, k, mode=mode, metric="euclidean", include_self=True)
    adj = A.toarray()
    adj_n = norm_adj(adj)
    return adj, adj_n


def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output


def normalize(X, filter_min=True, norm_cell=True, log=True, top_genes = 2500, scale=True):

    adata = AnnData(X=X)

    if filter_min:
        sc.pp.filter_genes(adata, min_counts=5)
        sc.pp.filter_cells(adata, min_counts=5)

    if norm_cell:
        sc.pp.normalize_per_cell(adata)

    if log:
        sc.pp.log1p(adata)

    if top_genes:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=top_genes)
        adata = adata[:, adata.var.highly_variable]

    if scale:
        sc.pp.scale(adata)

    return adata.X
