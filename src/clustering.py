import warnings
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


def cluster_cells_spatio_biological(
    adata: AnnData,
    spatial_key: str = 'spatial',
    use_rep: str = 'X_pca',
    resolution: float = 1.0,
    spatial_weight: float = 1.0,
    bio_weight: float = 1.0,
) -> np.ndarray:
    """
    Cluster cells into contiguous mesoregions using spatial coordinates and a
    shared biological embedding (e.g. Harmony-corrected joint PCA).

    Uses average linkage agglomerative clustering constrained to the Delaunay
    spatial graph.  Average linkage — unlike Ward — is well-defined under
    connectivity constraints because it only considers feasible (adjacent)
    merges and does not rely on a global inertia criterion that the constraint
    may render unreachable.

    Weights are dimension-normalised so that spatial_weight = bio_weight = 1.0
    gives each modality exactly equal contribution to the average-linkage merge
    cost regardless of how many PCA components are used.

    Args:
        adata:          AnnData object.
        spatial_key:    Key in adata.obsm storing 2-D spatial coordinates.
        use_rep:        Key in adata.obsm storing the shared biological
                        embedding (written by compute_shared_biological_embedding).
        resolution:     Scales the number of clusters linearly, analogous to
                        the resolution parameter in Leiden/Louvain.
        spatial_weight: Contribution of spatial modality after dimension
                        normalisation.  Raise to produce more compact,
                        spatially homogeneous clusters.
        bio_weight:     Contribution of biological modality after dimension
                        normalisation.  Raise to respect expression boundaries
                        more strongly.

    Returns:
        Integer cluster labels, shape (N,), ranging 0 .. C-1.
    """
    # ------------------------------------------------------------------ #
    # Input validation                                                     #
    # ------------------------------------------------------------------ #
    if spatial_key not in adata.obsm:
        raise ValueError(
            f"spatial_key '{spatial_key}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
    if use_rep not in adata.obsm:
        raise ValueError(
            f"use_rep '{use_rep}' not found in adata.obsm. "
            f"Run compute_shared_biological_embedding() first, or set "
            f"use_rep to an existing key. Available keys: {list(adata.obsm.keys())}"
        )

    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    bio_features = np.asarray(adata.obsm[use_rep], dtype=np.float64)
    n_cells = coords.shape[0]

    if not np.all(np.isfinite(bio_features)):
        raise ValueError(
            f"'{use_rep}' contains NaN or Inf values. "
            "Re-run the embedding step or impute missing values before clustering."
        )

    # ------------------------------------------------------------------- #
    # 1. Standardise and dimension-normalise both modalities              #
    #                                                                     #
    # After StandardScaler each feature has unit variance.  With d_s = 2  #
    # spatial dims and d_b biological dims, the spatial modality          #
    # contributes sum-of-squares ≈ d_s per cell and bio ≈ d_b.  Dividing  #
    # by sqrt(d) makes each modality contribute exactly (weight)^2 to the #
    # average-linkage distance, independent of dimensionality.            #
    # ------------------------------------------------------------------- #
    d_s = coords.shape[1]        # always 2 for 2-D spatial coordinates
    d_b = bio_features.shape[1]  # number of PCA components

    coords_scaled = StandardScaler().fit_transform(coords)
    bio_scaled    = StandardScaler().fit_transform(bio_features)

    coords_normed = coords_scaled * (spatial_weight / np.sqrt(d_s))
    bio_normed    = bio_scaled    * (bio_weight    / np.sqrt(d_b))

    X_combined = np.hstack([coords_normed, bio_normed])

    # ------------------------------------------------------------------ #
    # 2. Build Delaunay spatial connectivity graph                       #
    # ------------------------------------------------------------------ #
    edges = build_spatial_graph(coords)

    if not edges:
        return np.zeros(n_cells, dtype=int)

    row = np.array([e[0] for e in edges] + [e[1] for e in edges])
    col = np.array([e[1] for e in edges] + [e[0] for e in edges])
    data = np.ones(len(row))
    connectivity = sp.coo_matrix((data, (row, col)), shape=(n_cells, n_cells))

    # ------------------------------------------------------------------ #
    # 3. Target cluster count                                            #
    # ------------------------------------------------------------------ #
    n_clusters_target = max(2, min(
        n_cells - 1,                              # hard upper bound
        int((n_cells / 200.0) * resolution)
    ))

    # ------------------------------------------------------------------ #
    # 4. Spatially-constrained agglomerative clustering (ward linkage)   #
    # ------------------------------------------------------------------ #
    agg = AgglomerativeClustering(
        n_clusters=n_clusters_target,
        linkage='ward',           # correct under connectivity constraints
        connectivity=connectivity,
        metric='euclidean',
    )
    labels = agg.fit_predict(X_combined)

    return labels