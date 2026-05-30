import ot
import torch
import numpy as np

from anndata import AnnData
from numpy.typing import NDArray
from typing import Optional, Tuple, Union
from scipy.spatial import cKDTree
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

from .clustering import cluster_cells_spatial
from .hierarchical import (
    build_slice_cluster_cache,
    compute_cluster_feature_costs,
    compute_cluster_structural_matrix,
    extract_continuous_macro_section,
    run_coarse_partial_fgw,
)
from .utils import (
    select_backend,
    fused_gromov_wasserstein_incent,
    to_dense_array,
    extract_data_matrix,
    jensenshannon_divergence_backend,
    to_backend
)
from .visualize import (
    visualize_clustered_slices,
    visualize_cluster_mapping,
    visualize_initial_connected_component,
    visualize_selected_anchors,
    visualize_global_overlap_projection,
)


def hierarchical_pairwise_align(
    sliceA: AnnData,
    sliceB: AnnData,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float = 0.75,
    numItermax: int = 100000,
    use_gpu: bool = True,
    resolution: float = 1.0,
    spatial_key: str = "spatial",
    use_rep: Optional[str] = "X_pca",
    label_key: str = "cell_type_annot",
    visualize_clusters: bool = True,
    verbose: bool = False,
    **kwargs
):
    """
    Performs Hierarchical OT by clustering cells into mesoregions, aligning clusters with Partial FGW,
    and then restricting the cell-level OT matchings to the aligned blocks.
    
    Returns the cell-level alignment pi.
    """
    print("--- [HOT] Step 1: Clustering Cells into Mesoregions ---")
    labelsA = cluster_cells_spatial(sliceA, spatial_key=spatial_key, resolution=resolution)
    labelsB = cluster_cells_spatial(sliceB, spatial_key=spatial_key, resolution=resolution)

    # Pre-cache global cell types for cluster structure alignment
    all_types = np.array(sorted(set(sliceA.obs[label_key].astype(str)) | set(sliceB.obs[label_key].astype(str))), dtype=str)

    print(f"Slice A: {len(np.unique(labelsA))} clusters")
    print(f"Slice B: {len(np.unique(labelsB))} clusters")
    
    if visualize_clusters:
        visualize_clustered_slices(sliceA, sliceB, labelsA, labelsB, spatial_key=spatial_key)
    
    print("--- [HOT] Step 2: Extracting Cluster Features ---")
    cache_A = build_slice_cluster_cache(
        sliceA,
        labelsA,
        spatial_key=spatial_key,
        feature_key=use_rep,
        label_key=label_key,
        all_types=all_types,
    )
    cache_B = build_slice_cluster_cache(
        sliceB,
        labelsB,
        spatial_key=spatial_key,
        feature_key=use_rep,
        label_key=label_key,
        all_types=all_types,
    )
    p_A, centroidsA, mu_exprA, mu_structA = (
        cache_A.masses,
        cache_A.centroids,
        cache_A.mu_expr,
        cache_A.mu_struct,
    )
    p_B, centroidsB, mu_exprB, mu_structB= (
        cache_B.masses,
        cache_B.centroids,
        cache_B.mu_expr,
        cache_B.mu_struct
    )
    
    print("--- [HOT] Step 3: Compute Cluster Costs and Structures ---")
    M_cluster = compute_cluster_feature_costs(mu_exprA, mu_structA, mu_exprB, mu_structB, delta=delta)
    C_A = compute_cluster_structural_matrix(centroidsA)
    C_B = compute_cluster_structural_matrix(centroidsB)
    
    print("--- [HOT] Step 4: Run Coarse Partial FGW ---")
    Pi_cluster = run_coarse_partial_fgw(M_cluster, C_A, C_B, p_A, p_B, alpha=alpha)
    
    if visualize_clusters:
        visualize_cluster_mapping(centroidsA, centroidsB, Pi_cluster)

    # We now prepare the injection into standard cell-level pairwise_align
    print("--- [HOT] Step 5: Extract Continuous Macro Sections ---")
    macro_section = extract_continuous_macro_section(
        sliceA,
        sliceB,
        labelsA,
        labelsB,
        Pi_cluster,
        spatial_key=spatial_key,
        label_key=label_key,
        cluster_cache_A=cache_A,
        cluster_cache_B=cache_B,
    )
    if not macro_section.ok:
        raise ValueError(
            "Hierarchical alignment aborted: no trustworthy overlapping macro-component "
            f"was identified. Reason: {macro_section.reason}."
        )

    idx_A = macro_section.idx_A
    idx_B = macro_section.idx_B
    dist_A = macro_section.dist_A
    dist_B = macro_section.dist_B
    initial_idx_A = macro_section.initial_idx_A
    initial_idx_B = macro_section.initial_idx_B
    if macro_section.ambiguous:
        if macro_section.diagnostics.get("macro_hypothesis_ambiguity_detected", False):
            print(
                "[HOT] Warning: the top expanded macro hypotheses remained ambiguous. "
                f"Evidence ratio={macro_section.diagnostics.get('macro_hypothesis_evidence_ratio')}."
            )
        else:
            print(
                "[HOT] Warning: the initial macro seed family was ambiguous. "
                f"Evidence ratio={macro_section.diagnostics.get('seed_evidence_ratio')}."
            )
    if macro_section.alternative_hypotheses:
        print(
            "[HOT] Stored competing macro-overlap hypotheses: "
            f"{len(macro_section.alternative_hypotheses)}. "
            f"Winner came from seed trial {macro_section.diagnostics.get('selected_seed_trial_rank', 1)}."
        )
    
    print(f"Selected {len(idx_A)}/{sliceA.shape[0]} cells from A, {len(idx_B)}/{sliceB.shape[0]} cells from B.")

    if visualize_clusters:
        visualize_initial_connected_component(
            sliceA,
            sliceB,
            initial_idx_A,
            initial_idx_B,
            spatial_key=spatial_key
        )
        visualize_selected_anchors(sliceA, sliceB, idx_A, idx_B, spatial_key=spatial_key, dist_A=dist_A, dist_B=dist_B)

    print("--- [HOT] Step 6: Synthesizing Cell-Level Footprint from Macro Clusters ---")
    pi_full = np.zeros((sliceA.shape[0], sliceB.shape[0]), dtype=np.float64)
    
    if len(idx_A) > 0 and len(idx_B) > 0:
        # Restrict labels to the matched core
        core_labels_A = labelsA[idx_A]
        core_labels_B = labelsB[idx_B]
        
        for cA in range(Pi_cluster.shape[0]):
            for cB in range(Pi_cluster.shape[1]):
                mass = Pi_cluster[cA, cB]
                if mass > 0:
                    # Find cells in the core that belong to these clusters
                    cells_A = idx_A[core_labels_A == cA]
                    cells_B = idx_B[core_labels_B == cB]
                    
                    if len(cells_A) > 0 and len(cells_B) > 0:
                        # Distribute mass uniformly across the block
                        block_mass = mass / (len(cells_A) * len(cells_B))
                        grid_A, grid_B = np.ix_(cells_A, cells_B)
                        pi_full[grid_A, grid_B] += block_mass

    print("--- [HOT] Step 7: Global Refinement via Overlap Projection ---")
    from .visualize import stack_slices_pairwise
    try:
        # 1. Geometrically align full slices using the partial block solution
        aligned_slices = stack_slices_pairwise([sliceA, sliceB], [pi_full], output_params=False)
        coords_A_aligned = np.asarray(aligned_slices[0].obsm[spatial_key])
        coords_B_aligned = np.asarray(aligned_slices[1].obsm[spatial_key])
        
        # 2. Determine overlapping regions based on Morphological Rasterization (Exact Shadow)
        # Replaces soft radius (KDTree) or rigid boundaries (Convex Hull) with a grid footprint
        # capturing the actual boundaries, contours, and internal holes perfectly.
        s_A = estimate_characteristic_spacing(sliceA, spatial_key=spatial_key)
        s_B = estimate_characteristic_spacing(sliceB, spatial_key=spatial_key)
        grid_size = max(s_A, s_B) * 2.0  # Granularity is roughly 2 cell spaces wide
        
        # Compute common coordinate boundaries
        min_coords = np.minimum(coords_A_aligned.min(axis=0), coords_B_aligned.min(axis=0))

        # Convert continuous coordinates into discrete grid indices
        def to_grid(coords):
            return np.maximum(0, np.floor((coords - min_coords) / grid_size).astype(int))
            
        grid_A = to_grid(coords_A_aligned)
        grid_B = to_grid(coords_B_aligned)
        
        # Build 2D occupancy maps
        max_idx_A = grid_A.max(axis=0)
        max_idx_B = grid_B.max(axis=0)
        grid_bounds = np.maximum(max_idx_A, max_idx_B) + 5  # pad for safety
        
        from scipy.ndimage import binary_dilation
        mask_A = np.zeros(grid_bounds, dtype=bool)
        mask_B = np.zeros(grid_bounds, dtype=bool)
        
        mask_A[tuple(grid_A.T)] = True
        mask_B[tuple(grid_B.T)] = True
        
        # Dilate footprint slightly to connect cellular gaps mathematically without bloating the tissue
        mask_A_dilated = binary_dilation(mask_A, iterations=1)
        mask_B_dilated = binary_dilation(mask_B, iterations=1)
        
        # The exact, topological shadow intersection of both tissues
        overlap_mask = mask_A_dilated & mask_B_dilated
        
        overlap_mask_A = overlap_mask[tuple(grid_A.T)]
        overlap_mask_B = overlap_mask[tuple(grid_B.T)]
        
        # Fallback if overlap vanishes perfectly
        if not np.any(overlap_mask_A): overlap_mask_A[:] = True
        if not np.any(overlap_mask_B): overlap_mask_B[:] = True
        
        # 3. Supplying ONLY the shadow slices to the final OT pipeline
        # Within the shadow, apply decaying weights from the previously matched biological core
        idx_A_shadow = np.where(overlap_mask_A)[0]
        idx_B_shadow = np.where(overlap_mask_B)[0]
        
        sliceA_shadow = sliceA[idx_A_shadow].copy()
        sliceB_shadow = sliceB[idx_B_shadow].copy()
        
        # Scaling the decay dynamically based on the width of the shadow
        sigma_A = max(1e-5, np.max(dist_A[idx_A_shadow]) / 2.0)
        sigma_B = max(1e-5, np.max(dist_B[idx_B_shadow]) / 2.0)
        
        weight_A_shadow = np.exp(- (dist_A[idx_A_shadow]**2) / (2 * sigma_A**2))
        weight_B_shadow = np.exp(- (dist_B[idx_B_shadow]**2) / (2 * sigma_B**2))

        G_init_shadow = np.outer(weight_A_shadow, weight_B_shadow)
        G_init_sum = np.sum(G_init_shadow)
        if G_init_sum > 0:
            G_init_shadow /= G_init_sum

        if visualize_clusters:
                visualize_global_overlap_projection(
                    sliceA=sliceA,
                    sliceB=sliceB,
                    idx_A_shadow=idx_A_shadow,
                    idx_B_shadow=idx_B_shadow,
                    weight_A_shadow=weight_A_shadow,
                    weight_B_shadow=weight_B_shadow,
                    spatial_key=spatial_key
                )

        print(f"--- [HOT] Step 8: Executing Final Base OT on Shadow Portions (A: {len(idx_A_shadow)}, B: {len(idx_B_shadow)}) ---")
        pi_shadow_final = pairwise_align(
            sliceA=sliceA_shadow,
            sliceB=sliceB_shadow,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            G_init=G_init_shadow,
            numItermax=numItermax,
            use_gpu=use_gpu,
            verbose=verbose,
            **kwargs
        )
        
        # Inject the dense output explicitly into the sparse global matrix footprint
        pi_full_final = np.zeros((sliceA.shape[0], sliceB.shape[0]), dtype=np.float64)
        grid_A, grid_B = np.ix_(idx_A_shadow, idx_B_shadow)
        pi_full_final[grid_A, grid_B] = pi_shadow_final

        return pi_full_final
        
    except Exception as e:
        print(f"Global refinement failed: {e}. Returning block-restricted pi_full.")
        return pi_full


def align_multiple_slices(
    slices: list[AnnData],
    spatial_key: str = "spatial",
    **kwargs
) -> list[NDArray]:
    """
    Aligns a stack of N spatial transcriptomics slices sequentially (0 <- 1 <- 2 ... <- N).
    
    Args:
        slices: A list of N AnnData objects representing adjacent tissue slices.
        spatial_key: The key in obsm for the spatial coordinates.
        visualize_stack: Whether to plot the multi-slice 3D spatial alignment.
        z_spacing: Distance parameter artificial height injected between slices.
        **kwargs: Arguments to pass down to hierarchical_pairwise_align.
        
    Returns:
        aligned_slices: List of N AnnData objects with updated spatial coordinates aligned to slice 0.
        pi_matrices: List of N-1 transport matrices mapping each slice i to i+1.
    """
    if len(slices) < 2:
        raise ValueError("At least 2 slices are required for alignment.")
    
    pi_matrices = []
    
    print(f"Starting Multi-Slice Alignment for {len(slices)} slices...")
    
    # Step 1: Compute OT matchings for all consecutive pairs
    for i in range(len(slices) - 1):
        print(f"\n{'='*40}")
        print(f"Aligning Pair {i} and {i+1}")
        print(f"{'='*40}")
        
        pi_pair = hierarchical_pairwise_align(
            sliceA=slices[i], 
            sliceB=slices[i+1], 
            spatial_key=spatial_key,
            **kwargs
        )
        pi_matrices.append(pi_pair)
    
    return pi_matrices


def pairwise_align(
    sliceA: AnnData,
    sliceB: AnnData,
    alpha: float,
    beta: float,
    gamma: float,
    radius: Optional[float] = None,
    use_rep: Optional[str] = None,
    G_init = None,
    a_distribution = None,
    b_distribution = None,
    numItermax: int = 6000,
    use_gpu: bool = False,
    data_type = np.float32,
    epsilon: float = 1e-6,
    verbose: bool = True,
    gpu_verbose: bool = True,
    **kwargs) -> Union[NDArray[np.floating], Tuple[NDArray[np.floating], float, float, float, float]]:
    """

    This method is written by Anup Bhowmik, CSE, BUET

    Calculates and returns optimal alignment of two slices of single cell MERFISH data. 
    
    Args:
        sliceA: Slice A to align.
        sliceB: Slice B to align.
        alpha:  weight for spatial distance
        gamma: weight for gene expression distance (JSD)
        beta: weight for cell type one hot encoding
        radius: radius for cellular neighborhood

        dissimilarity: Expression dissimilarity measure: ``'kl'`` or ``'euclidean'``.
        use_rep: If ``None``, uses ``slice.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``slice.obsm[use_rep]``.
        G_init (array-like, optional): Initial mapping to be used in FGW-OT, otherwise default is uniform mapping.
        a_distribution (array-like, optional): Distribution of sliceA spots, otherwise default is uniform.
        b_distribution (array-like, optional): Distribution of sliceB spots, otherwise default is uniform.
        numItermax: Max number of iterations during FGW-OT.
        norm: If ``True``, scales spatial distances such that neighboring spots are at distance 1. Otherwise, spatial distances remain unchanged.
        backend: Type of backend to run calculations. For list of backends available on system: ``ot.backend.get_backend_list()``.
        data_type: Data type for backend tensors. Default is float32.
        return_obj: If ``True``, additionally returns objective function output of FGW-OT and cell-type matching metrics.
        verbose: If ``True``, FGW-OT is verbose.
        gpu_verbose: If ``True``, print whether gpu is being used to user.
   
    Returns:
        - Alignment of spots (pi).

        If ``return_obj = True``, additionally returns:
        
        - initial_obj_neighbor, initial_obj_gene, final_obj_neighbor, final_obj_gene: Objective metrics
        - initial_cell_type_match, final_cell_type_match: Cell-type matching percentages 
    """
    
    # Determine if gpu or cpu is being used
    use_gpu, nx = select_backend(use_gpu=use_gpu, gpu_verbose=gpu_verbose)
    
    
    # check if slices are valid
    for s in [sliceA, sliceB]:
        if not len(s):
            raise ValueError(f"Found empty `AnnData`:\n{s}.")   
    
    # ────────────────────── Calculate spatial distances ──────────────────────
    D_A, D_B = calculate_spatial_distance(sliceA, sliceB, nx, data_type=data_type, eps=epsilon)
    

    # ────────────────────── Calculate gene expression dissimilarity ──────────────────────
    cosine_dist_gene_expr = calculate_gene_expression_cosine_distance(sliceA, sliceB, use_rep, eps=epsilon)


    # ────────────────────── Calculate cell-type mismatch penalty ──────────────────────
    cell_type_mismatch = calculate_cell_type_mismatch(sliceA, sliceB)


    # Combine gene expression dissimilarity and cell-type mismatch penalty into a single cost matrix M1
    M1_combined = (1 - beta - gamma) * cosine_dist_gene_expr + beta * cell_type_mismatch
    M1 = to_backend(M1_combined, nx, data_type=data_type)


    # ────────────────────── Calculate neighborhood dissimilarity ──────────────────────
    js_dist_neighborhood = calculate_neighborhood_dissimilarity(
        sliceA,
        sliceB,
        radius=radius,                 # optional; if None multiscale radii are estimated
        nx=nx,
        data_type=data_type,
        eps=epsilon,
        radii=None,                    # or pass an explicit list, e.g. [20, 35, 50]
        radius_k=3,
        radius_multipliers=(2.5, 4.0, 5.0),
        n_shells=3,
        harmonics=(0, 1, 2),
        harmonic_weights=None,
        distance_decay="linear",
        include_self=False,
    )
    M2 = to_backend(js_dist_neighborhood, nx, data_type=data_type)

    # init distributions
    if a_distribution is None:
        a = nx.from_numpy(np.ones(sliceA.shape[0], dtype=np.float64) / sliceA.shape[0])
    else:
        a = nx.from_numpy(a_distribution)
        
    if b_distribution is None:
        b = nx.from_numpy(np.ones(sliceB.shape[0], dtype=np.float64) / sliceB.shape[0])
    else:
        b = nx.from_numpy(b_distribution)

    a = to_backend(a, nx, data_type=data_type)
    b = to_backend(b, nx, data_type=data_type)
    
    # Run OT
    if G_init is not None:
        G_init = to_backend(G_init, nx, data_type=data_type)

    pi = fused_gromov_wasserstein_incent(M1 + gamma * M2, D_A, D_B, a, b, G_init = G_init, alpha= alpha, numItermax=numItermax, verbose=verbose, **kwargs)
    pi = nx.to_numpy(pi)

    if isinstance(nx, ot.backend.TorchBackend):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pi


def estimate_characteristic_spacing(adata, k=3, spatial_key="spatial"):
    """
    Robust local spacing estimate: median distance to the k-th nearest neighbor.
    Useful for choosing radii that scale with local cell density.
    """
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    n = coords.shape[0]
    if n < 2:
        return 1.0

    k_eff = min(k + 1, n)  # +1 because nearest neighbor includes self at distance 0
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=k_eff)

    kth = dists[:, -1]
    kth = kth[np.isfinite(kth) & (kth > 0)]
    if kth.size == 0:
        return 1.0
    return float(np.median(kth))


def equal_area_shell_edges(radius, n_shells):
    """
    Equal-area shells reduce the trivial bias that outer shells cover more area.
    """
    return radius * np.sqrt(np.linspace(0.0, 1.0, n_shells + 1))


def distance_weights(dist, radius, mode="linear", sigma=None):
    """
    Distance weighting for neighbors inside a radius.
    """
    if mode is None or mode == "uniform":
        return np.ones_like(dist, dtype=np.float64)

    if mode == "linear":
        return np.maximum(0.0, 1.0 - dist / radius)

    if mode == "gaussian":
        s = radius / 2.0 if sigma is None else float(sigma)
        if s <= 0:
            raise ValueError("sigma must be > 0")
        return np.exp(-(dist ** 2) / (2.0 * s * s))

    raise ValueError("distance_decay must be one of: None, 'uniform', 'linear', 'gaussian'")


def neighborhood_distribution_fourier(
    adata,
    radius,
    cell_types=None,
    n_shells=3,
    shell_edges=None,
    harmonics=(0, 1, 2),
    harmonic_weights=None,
    distance_decay="linear",
    sigma=None,
    include_self=False,
    area_normalize=True,
    add_empty_bin=True,
    l1_normalize=True,
    dtype=np.float32,
    spatial_key="spatial",
    label_key="cell_type_annot",
    return_metadata=False,
):
    """
    Rotation- and reflection-invariant neighborhood descriptor.

    For each focal cell, each cell type, and each radial shell:
      m=0 -> abundance
      m=1 -> polarity magnitude
      m=2 -> bilateral / opposite-half anisotropy magnitude

    Nonzero harmonics are represented by their complex magnitudes rather than
    phase-resolved real/imaginary parts. This yields descriptors that remain
    stable under arbitrary in-plane rotation and reflection, which is essential
    for partial tissue alignment before orientation has been resolved.

    Output is nonnegative and suitable for Jensen-Shannon after normalization.
    """
    if radius <= 0:
        raise ValueError("radius must be > 0")

    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    labels = adata.obs[label_key].astype(str).to_numpy()

    if cell_types is None:
        cell_types = np.array(sorted(np.unique(labels)), dtype=str)
    else:
        cell_types = np.array(cell_types, dtype=str)

    label_to_idx = {ct: i for i, ct in enumerate(cell_types)}
    missing = sorted(set(labels) - set(cell_types))
    if missing:
        raise ValueError(f"Labels missing from cell_types: {missing}")

    label_idx = np.array([label_to_idx[x] for x in labels], dtype=np.int32)

    harmonics = tuple(sorted(set(int(h) for h in harmonics)))
    if any(h < 0 for h in harmonics):
        raise ValueError("harmonics must be non-negative integers")
    if 0 not in harmonics:
        harmonics = (0,) + harmonics

    if harmonic_weights is None:
        harmonic_weights = {}
    else:
        harmonic_weights = {int(k): float(v) for k, v in harmonic_weights.items()}

    if shell_edges is None:
        shell_edges = equal_area_shell_edges(radius, n_shells)
    else:
        shell_edges = np.asarray(shell_edges, dtype=np.float64)
        if not np.isclose(shell_edges[0], 0.0):
            raise ValueError("shell_edges must start at 0")
        if not np.isclose(shell_edges[-1], radius):
            raise ValueError("shell_edges must end at radius")
        if np.any(np.diff(shell_edges) <= 0):
            raise ValueError("shell_edges must be strictly increasing")

    n_cells = coords.shape[0]
    n_types = len(cell_types)
    n_shells_eff = len(shell_edges) - 1
    
    n_harm = len(harmonics)

    shell_areas = np.pi * (shell_edges[1:] ** 2 - shell_edges[:-1] ** 2)

    # group index = type * n_shells + shell
    n_groups = n_types * n_shells_eff
    group_shell_idx = np.tile(np.arange(n_shells_eff), n_types)
    group_area = shell_areas[group_shell_idx]

    tree = cKDTree(coords)
    neighbor_lists = tree.query_ball_point(coords, r=radius)

    n_core = n_groups * n_harm
    n_total = n_core + (1 if add_empty_bin else 0)
    features = np.zeros((n_cells, n_total), dtype=np.float64)

    for i, nbr in enumerate(neighbor_lists):
        nbr = np.asarray(nbr, dtype=np.int32)

        if not include_self:
            nbr = nbr[nbr != i]

        if nbr.size == 0:
            if add_empty_bin:
                features[i, -1] = 1.0
            continue

        rel = coords[nbr] - coords[i]
        dist = np.linalg.norm(rel, axis=1)
        theta = np.arctan2(rel[:, 1], rel[:, 0])

        shell_idx = np.searchsorted(shell_edges[1:], dist, side="left")
        valid = (shell_idx >= 0) & (shell_idx < n_shells_eff)

        if not np.all(valid):
            nbr = nbr[valid]
            dist = dist[valid]
            theta = theta[valid]
            shell_idx = shell_idx[valid]

        if nbr.size == 0:
            if add_empty_bin:
                features[i, -1] = 1.0
            continue

        w = distance_weights(dist, radius=radius, mode=distance_decay, sigma=sigma)
        group_idx = label_idx[nbr] * n_shells_eff + shell_idx

        local = np.zeros((n_groups, n_harm), dtype=np.float64)

        h_pos = 0
        for m in harmonics:
            if m == 0:
                mag = np.bincount(group_idx, weights=w, minlength=n_groups).astype(np.float64)
                if area_normalize:
                    mag = mag / np.maximum(group_area, 1e-12)
                mag *= harmonic_weights.get(m, 1.0)
                local[:, h_pos] = mag
                h_pos += 1
            else:
                ang = m * theta
                real = np.bincount(group_idx, weights=w * np.cos(ang), minlength=n_groups)
                imag = np.bincount(group_idx, weights=w * np.sin(ang), minlength=n_groups)

                magnitude = np.sqrt(real ** 2 + imag ** 2)
                if area_normalize:
                    magnitude = magnitude / np.maximum(group_area, 1e-12)

                magnitude *= harmonic_weights.get(m, 1.0)

                local[:, h_pos] = magnitude
                h_pos += 1

        flat = local.reshape(-1)

        if flat.sum() == 0:
            if add_empty_bin:
                features[i, -1] = 1.0
        else:
            if add_empty_bin:
                features[i, :-1] = flat
            else:
                features[i] = flat

    if l1_normalize:
        row_sums = features.sum(axis=1, keepdims=True)
        nz = row_sums[:, 0] > 0
        features[nz] /= row_sums[nz]

    features = features.astype(dtype, copy=False)

    if not return_metadata:
        return features

    metadata = {
        "cell_types": cell_types,
        "shell_edges": shell_edges,
        "harmonics": harmonics,
        "feature_shape": (n_types, n_shells_eff, len(harmonics)),
    }
    return features, metadata


def default_radii_from_spacing(sliceA, sliceB, k=3, multipliers=(2.5, 4.0, 5.0), spatial_key="spatial"):
    sA = estimate_characteristic_spacing(sliceA, k=k, spatial_key=spatial_key)
    sB = estimate_characteristic_spacing(sliceB, k=k, spatial_key=spatial_key)
    base = max(sA, sB)
    return [m * base for m in multipliers]


def neighborhood_distribution_multiscale(
    adata,
    radii,
    cell_types=None,
    n_shells=3,
    harmonics=(0, 1, 2),
    harmonic_weights=None,
    distance_decay="linear",
    sigma=None,
    include_self=False,
    area_normalize=True,
    add_empty_bin_per_scale=False,
    l1_normalize_within_scale=True,
    final_l1_normalize=True,
    dtype=np.float32,
    spatial_key="spatial",
    label_key="cell_type_annot",
    return_metadata=False,
):
    """
    Concatenate rotation- and reflection-invariant descriptors across multiple radii.
    """
    radii = [float(r) for r in radii]
    if any(r <= 0 for r in radii):
        raise ValueError("all radii must be > 0")

    blocks = []
    meta_blocks = []

    for r in radii:
        feat, meta = neighborhood_distribution_fourier(
            adata=adata,
            radius=r,
            cell_types=cell_types,
            n_shells=n_shells,
            harmonics=harmonics,
            harmonic_weights=harmonic_weights,
            distance_decay=distance_decay,
            sigma=sigma,
            include_self=include_self,
            area_normalize=area_normalize,
            add_empty_bin=add_empty_bin_per_scale,
            l1_normalize=l1_normalize_within_scale,
            dtype=np.float64,
            spatial_key=spatial_key,
            label_key=label_key,
            return_metadata=True,
        )
        blocks.append(feat)
        meta_blocks.append({"radius": r, **meta})

    X = np.concatenate(blocks, axis=1)

    if final_l1_normalize:
        row_sums = X.sum(axis=1, keepdims=True)
        nz = row_sums[:, 0] > 0
        X[nz] /= row_sums[nz]

    X = X.astype(dtype, copy=False)

    if not return_metadata:
        return X

    return X, {"scales": meta_blocks}


def calculate_spatial_distance(sliceA, sliceB, nx, data_type=np.float32, spatial_key = 'spatial', eps=1e-8, norm_k=3):
    """
    Calculate spatial distance between cells in a slice, normalized by robust local spacing.

    Args:
        sliceA: First slice for which to calculate spatial distance.
        sliceB: Second slice for which to calculate spatial distance.
        nx: Backend to use for calculations.
        data_type: Data type for backend tensors.
        spatial_key: Key for the spatial coordinates.
        eps: Small constant to avoid division by zero.
        norm_k: Kth neighbor to use for characteristic spacing estimation.
    Returns:
    D_A, D_B: Pairwise spatial distance matrices.
    """
    
    print("Calculating spatial distance between cells in slice A and slice B")

    coordinates_A = np.asarray(sliceA.obsm[spatial_key], dtype=np.float64)
    coordinates_B = np.asarray(sliceB.obsm[spatial_key], dtype=np.float64)

    D_A = euclidean_distances(coordinates_A, coordinates_A)
    D_B = euclidean_distances(coordinates_B, coordinates_B)

    # Normalize by local characteristic spacing instead of global max tissue diameter
    scale_A = estimate_characteristic_spacing(sliceA, k=norm_k, spatial_key=spatial_key)
    scale_B = estimate_characteristic_spacing(sliceB, k=norm_k, spatial_key=spatial_key)
    scale = max(scale_A, scale_B, eps)

    D_A = D_A / scale
    D_B = D_B / scale

    D_A = to_backend(D_A, nx, data_type=data_type)
    D_B = to_backend(D_B, nx, data_type=data_type)

    return D_A, D_B


def calculate_gene_expression_cosine_distance(sliceA, sliceB, use_rep, eps = 1e-6):
    """
    Calculate cosine distance between gene expression profiles of slice A and slice B.
    
    Args:
    sliceA: First slice.
    sliceB: Second slice.
    use_rep: If ``None``, uses ``slice.X`` to calculate dissimilarity
                between spots, otherwise uses the representation given by ``slice.obsm[use_rep]``.
    eps: Small constant to add to data matrices to avoid zero vectors.
    
    Returns:
    cosine_dist_gene_expr: Cosine distance matrix between gene expression profiles of slice A and slice B.
    """
    
    print("Calculating cosine distance between gene expression profiles of slice A and slice B")
    
    # Extract and prepare data matrices for cosine distance calculation
    A_X = extract_data_matrix(sliceA, use_rep)
    B_X = extract_data_matrix(sliceB, use_rep)

    # Convert to dense arrays and add small constant to avoid zero vectors
    A_X = to_dense_array(A_X) + eps
    B_X = to_dense_array(B_X) + eps

    # Use sklearn's optimized and numerically stable cosine_distances
    cosine_dist_gene_expr = cosine_distances(A_X, B_X)

    return cosine_dist_gene_expr


def calculate_cell_type_mismatch(sliceA, sliceB):
    """
    Calculate the cell-type mismatch penalty between two slices.

    Args:
        sliceA: First slice.
        sliceB: Second slice.

    Returns:
        cell_type_mismatch: Binary matrix indicating cell-type mismatches.
    """

    _lab_A = np.asarray(sliceA.obs['cell_type_annot'].values)
    _lab_B = np.asarray(sliceB.obs['cell_type_annot'].values)

    cell_type_mismatch = (_lab_A[:, None] != _lab_B[None, :]).astype(np.float64)

    return cell_type_mismatch


def calculate_neighborhood_dissimilarity(
    sliceA,
    sliceB,
    radius=None,
    nx=None,
    data_type=np.float32,
    eps=1e-8,
    radii=None,
    radius_k=3,
    radius_multipliers=(2.5, 4.0, 5.0),
    n_shells=3,
    harmonics=(0, 1, 2),
    harmonic_weights=None,
    distance_decay="linear",
    sigma=None,
    include_self=False,
    spatial_key="spatial",
    label_key="cell_type_annot",
):
    """
    Neighborhood dissimilarity using multiscale rotation- and reflection-invariant descriptors
    and proper re-normalization before Jensen-Shannon distance.
    """
    all_types = np.array(sorted(
        set(sliceA.obs[label_key].astype(str)) |
        set(sliceB.obs[label_key].astype(str))
    ), dtype=str)

    if radii is None:
        if radius is not None:
            radii = [float(radius)]
        else:
            radii = default_radii_from_spacing(
                sliceA,
                sliceB,
                k=radius_k,
                multipliers=radius_multipliers,
                spatial_key=spatial_key,
            )

    featA = neighborhood_distribution_multiscale(
        sliceA,
        radii=radii,
        cell_types=all_types,
        n_shells=n_shells,
        harmonics=harmonics,
        harmonic_weights=harmonic_weights,
        distance_decay=distance_decay,
        sigma=sigma,
        include_self=include_self,
        area_normalize=True,
        add_empty_bin_per_scale=False,
        l1_normalize_within_scale=True,
        final_l1_normalize=True,
        dtype=np.float32,
        spatial_key=spatial_key,
        label_key=label_key,
    )

    featB = neighborhood_distribution_multiscale(
        sliceB,
        radii=radii,
        cell_types=all_types,
        n_shells=n_shells,
        harmonics=harmonics,
        harmonic_weights=harmonic_weights,
        distance_decay=distance_decay,
        sigma=sigma,
        include_self=include_self,
        area_normalize=True,
        add_empty_bin_per_scale=False,
        l1_normalize_within_scale=True,
        final_l1_normalize=True,
        dtype=np.float32,
        spatial_key=spatial_key,
        label_key=label_key,
    )

    # Add epsilon, THEN renormalize.
    featA = featA + eps
    featB = featB + eps
    featA = featA / featA.sum(axis=1, keepdims=True)
    featB = featB / featB.sum(axis=1, keepdims=True)

    if nx is None:
        return featA, featB

    featA = to_backend(featA, nx, data_type=data_type)
    featB = to_backend(featB, nx, data_type=data_type)

    return jensenshannon_divergence_backend(featA, featB)
