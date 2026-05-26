import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay
from anndata import AnnData
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


def build_spatial_graph(coords: np.ndarray):
    """
    Builds a spatial connectivity graph (adjacency matrix/edge list).
    
    Args:
        coords: Spatial coordinates array shape (N, 2).
        
    Returns:
        edges: list of tuples (i, j).
        weights: corresponding edge weights (e.g. 1/distance, or 1.0).
    """
    n_cells = coords.shape[0]
    edges = []
    
    tri = Delaunay(coords)
    indptr, indices = tri.vertex_neighbor_vertices
    for i in range(n_cells):
        for j in indices[indptr[i]:indptr[i+1]]:
            if i < j:  # Avoid duplicates
                edges.append((i, j))
        
    return edges


def cluster_cells_spatial(adata: AnnData, spatial_key: str = 'spatial', resolution: float = 1.0) -> np.ndarray:
    """
    Clustering cells into contiguous supercells (mesoregions) using Spatially-Constrained Agglomerative Clustering.
    This guarantees 100% deterministic, mathematically fixed spatial regions without seed dependency.
    
    Args:
        adata: AnnData object.
        spatial_key: Key in adata.obsm storing spatial coordinates.
        resolution: Resolution parameter (scales number of clusters).
        
    Returns:
        Cluster labels from 0 to C-1.
    """
    coords = adata.obsm[spatial_key]
    n_cells = coords.shape[0]
    
    # 1. Build Spatial Graph
    edges = build_spatial_graph(coords)
    
    # 2. Create Connectivity Matrix
    if not edges:
        return np.zeros(n_cells, dtype=int)
        
    row = np.array([e[0] for e in edges] + [e[1] for e in edges])
    col = np.array([e[1] for e in edges] + [e[0] for e in edges])
    data = np.ones(len(row))
    connectivity = sp.coo_matrix((data, (row, col)), shape=(n_cells, n_cells))
    
    # 3. Determine Number of Clusters
    # Scale number of clusters based on total cells and resolution to mimic Leiden behavior
    n_clusters_target = max(2, int((n_cells / 200.0) * resolution))
    
    # 4. Spatially-Constrained Agglomerative Clustering
    agg = AgglomerativeClustering(
        n_clusters=n_clusters_target,
        linkage='ward',
        connectivity=connectivity
    )
    labels = agg.fit_predict(coords)

    return labels


def cluster_cells_spatio_biological(adata: AnnData, spatial_key: str = 'spatial', use_rep: str = 'X_pca', resolution: float = 1.0, spatial_weight: float = 1.0, bio_weight: float = 1.0) -> np.ndarray:
    """
    Clustering cells into mesoregions using BOTH spatial coordinates and biological features.
    This produces contiguous mesoregions (due to spatial connectivity) that group cells with similar 
    biological profiles, helping matched structures cleanly respect biological boundaries.
    
    Args:
        adata: AnnData object.
        spatial_key: Key in adata.obsm storing spatial coordinates.
        use_rep: Key in adata.obsm storing biological features (e.g. 'X_pca').
        resolution: Resolution parameter (scales number of clusters).
        spatial_weight: Importance of spatial coordinates vs biology.
        bio_weight: Importance of biological features.
        
    Returns:
        Cluster labels from 0 to C-1.
    """
    coords = adata.obsm[spatial_key]
    bio_features = adata.obsm[use_rep]
    n_cells = coords.shape[0]
    
    # 1. Standardize both modalities so they are functionally comparable
    coords_scaled = StandardScaler().fit_transform(coords)
    bio_scaled = StandardScaler().fit_transform(bio_features)
    
    # 2. Apply weights and concatenate into a single feature space
    X_combined = np.hstack([
        coords_scaled * spatial_weight,
        bio_scaled * bio_weight
    ])
    
    # 3. Build Spatial Graph (connectivity relies ONLY on unscaled physical space)
    edges = build_spatial_graph(coords)
    
    if not edges:
        return np.zeros(n_cells, dtype=int)
        
    # 4. Create Connectivity Matrix
    row = np.array([e[0] for e in edges] + [e[1] for e in edges])
    col = np.array([e[1] for e in edges] + [e[0] for e in edges])
    data = np.ones(len(row))
    connectivity = sp.coo_matrix((data, (row, col)), shape=(n_cells, n_cells))
    
    # 5. Spatially-Constrained Agglomerative Clustering using Combined Features
    n_clusters_target = max(2, int((n_cells / 200.0) * resolution))
    agg = AgglomerativeClustering(
        n_clusters=n_clusters_target,
        linkage='ward',
        connectivity=connectivity
    )
    labels = agg.fit_predict(X_combined)

    return labels

