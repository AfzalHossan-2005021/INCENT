"""
Spatially-constrained biological clustering for INCENT mesoregions.

This module provides deterministic, parameter-free spatial clustering based on 
Otsu's thresholding and Connected Components. 

When a trustworthy cell-type annotation is available in adata.obs (`use_celltype`), 
the connected components of the per-type Delaunay subgraph (filtered by automated 
Otsu threshold on edge lengths) become clusters. This provides a mathematically 
pure definition of a mesoregion that is stable even when tissue is truncated 
asymmetrically.

If the annotation is not available, the fallback mode uses biological Ward 
agglomerative clustering on gene-expression PCA (or any obsm representation), 
constrained by the spatial connectivity graph.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay
from anndata import AnnData
from sklearn.cluster import AgglomerativeClustering
from typing import Optional


# ============================================================================
# Spatial graph (unchanged from original)
# ============================================================================

def build_spatial_graph(coords: np.ndarray):
    """
    Builds a spatial connectivity graph (adjacency matrix/edge list).
    Uses a parameter-free Otsu's thresholding on edge lengths to automatically
    remove spurious long edges that bridge tissue gaps or concave boundaries.
    
    Args:
        coords: Spatial coordinates array shape (N, 2).

    Returns:
        edges: list of tuples (i, j).
    """
    n_cells = coords.shape[0]
    tri = Delaunay(coords)
    indptr, indices = tri.vertex_neighbor_vertices
    
    raw_edges = []
    lengths = []
    
    for i in range(n_cells):
        for j in indices[indptr[i]:indptr[i + 1]]:
            if i < j:  # Avoid duplicates
                d = np.linalg.norm(coords[i] - coords[j])
                raw_edges.append((i, j))
                lengths.append(d)

    if not lengths:
        return []
        
    lengths = np.array(lengths)
    
    # Parameter-free Otsu thresholding to drop spurious boundary-spanning Delaunay edges
    bins = min(100, len(lengths))
    hist, bin_edges = np.histogram(lengths, bins=bins)
    hist = hist / np.sum(hist)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    w1 = np.cumsum(hist)
    w2 = 1.0 - w1
    
    mu_cumsum = np.cumsum(hist * bin_centers)
    mu1 = mu_cumsum / np.maximum(w1, 1e-12)
    mu_total = mu_cumsum[-1]
    
    mu2 = (mu_total - mu_cumsum) / np.maximum(w2, 1e-12)
    
    variance_between = w1 * w2 * (mu1 - mu2) ** 2
    optimal_idx = np.argmax(variance_between)
    threshold = bin_centers[optimal_idx]

    edges = []
    for (i, j), length in zip(raw_edges, lengths):
        if length <= threshold:
            edges.append((i, j))

    return edges


from sklearn.preprocessing import normalize

# ============================================================================
# Feature-matrix selection
# ============================================================================

def _resolve_continuous_features(adata: AnnData, use_rep: Optional[str], spatial_key: str) -> np.ndarray:
    """
    Return the (n_cells, d) feature matrix for continuous expression / PCA.
    """
    if use_rep == "spatial":
        features = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    elif use_rep is not None:
        features = np.asarray(adata.obsm[use_rep], dtype=np.float64)
    elif "X_pca" in adata.obsm:
        features = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
    else:
        X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X, dtype=np.float64)
        if X.shape[1] > 50:
            X_centered = X - X.mean(axis=0, keepdims=True)
            U, S, _Vt = np.linalg.svd(X_centered, full_matrices=False)
            n_pc = min(30, U.shape[1])
            features = (U[:, :n_pc] * S[:n_pc]).astype(np.float64)
        else:
            features = X
    return features


# ============================================================================
# Ward Integration
# ============================================================================

def _fit_ward(features: np.ndarray, connectivity: sp.spmatrix, n_clusters: int) -> np.ndarray:
    agg = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward",
        connectivity=connectivity,
    )
    return agg.fit_predict(features)


def cluster_cells_spatial(
    adata: AnnData,
    spatial_key: str = "spatial",
    resolution: float = 1.0,
    use_rep: Optional[str] = None,
    use_celltype: Optional[str] = None,
    **kwargs
) -> np.ndarray:
    """
    Parameter-free, truncation-invariant biological clustering.
    
    This function uses Local Graph Convolution (Majority Voting) + Connected Components 
    to guarantee that the interior of truncated spatial slices maintains structurally 
    isomorphic macro-clusters, bypassing the fragmentation issues of rigid boundaries 
    without introducing the shift-variance of global objective functions (like Ward).
    """
    coords = np.asarray(adata.obsm[spatial_key])
    n_cells = coords.shape[0]

    # 1. Build parameter-free filtered spatial graph
    edges = build_spatial_graph(coords)
    
    if not edges:
        return np.zeros(n_cells, dtype=int)

    # 2. Resolve continuous biological features
    features = _resolve_continuous_features(adata, use_rep, spatial_key)

    if use_celltype is not None and use_celltype in adata.obs.columns:
        # 3. Create Continuous-Expression Weighted Edges (Cosine Similarity)
        # This parameter-free bounding acts as biological logic gates:
        # Negative sim -> 0.0 (Graph boundary cut between vastly different tissues)
        # Positive sim -> 0-1.0 (Gradient transition within related tissues)
        features_norm = normalize(features, norm='l2', axis=1)
        
        row, col, data = [], [], []
        for (i, j) in edges:
            sim = float(np.dot(features_norm[i], features_norm[j]))
            w = max(0.0, sim)
            row.extend([i, j])
            col.extend([j, i])
            data.extend([w, w])
            
        W = sp.coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsr()
        
        # Row-normalize to establish unbiased Markov Transition Matrix with self-loops
        W_self = W + sp.eye(n_cells, format='csr')
        W_norm = normalize(W_self, norm='l1', axis=1)
        
        # 4. Multi-modal 2-Hop Graph Convolution (Smooths out exact salt-and-pepper noise)
        series = adata.obs[use_celltype].astype("category")
        Y = np.zeros((n_cells, len(series.cat.categories)), dtype=np.float64)
        Y[np.arange(n_cells), series.cat.codes] = 1.0
        
        Y_smooth = W_norm.dot(Y)
        Y_smooth = W_norm.dot(Y_smooth) # 2nd Hop expands local receptive field
        
        smoothed_types = np.argmax(Y_smooth, axis=1)
        
        # 5. Extract strict Connected Components constrained purely by Smoothed Types
        nbrs = [[] for _ in range(n_cells)]
        for i, j in edges:
            if smoothed_types[i] == smoothed_types[j]:
                nbrs[i].append(j)
                nbrs[j].append(i)
                
        labels = -np.ones(n_cells, dtype=int)
        next_id = 0
        
        for i in range(n_cells):
            if labels[i] == -1:
                stack = [i]
                labels[i] = next_id
                while stack:
                    u = stack.pop()
                    for v in nbrs[u]:
                        if labels[v] == -1:
                            labels[v] = next_id
                            stack.append(v)
                next_id += 1
                
        _, final_labels = np.unique(labels, return_inverse=True)
        return final_labels.astype(int)

    # Fallback to standard Ward linkage if no cell types are provided
    row = np.array([e[0] for e in edges] + [e[1] for e in edges])
    col = np.array([e[1] for e in edges] + [e[0] for e in edges])
    data = np.ones(len(row), dtype=np.float64)
    connectivity = sp.coo_matrix((data, (row, col)), shape=(n_cells, n_cells))

    # A standard fixed scale when falling back to unsupervised features without parameter hints
    n_clusters_target = max(2, int((n_cells / 200.0) * resolution))

    labels = _fit_ward(features, connectivity, n_clusters_target).astype(int)
    _, final_labels = np.unique(labels, return_inverse=True)
    return final_labels.astype(int)