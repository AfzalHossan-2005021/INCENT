import ot
import torch
import numpy as np

from anndata import AnnData
from numpy.typing import NDArray
from typing import Optional, Tuple, Union
from scipy.spatial import cKDTree, Voronoi
from sklearn.metrics.pairwise import cosine_distances

from .clustering import cluster_cells_spatial
from .hierarchical import (
    build_slice_cluster_cache,
    compute_cluster_feature_costs,
    compute_cluster_structural_matrix,
    compute_pairwise_mutual_information_contribution,
    extract_continuous_macro_section,
    fit_weighted_rigid_transform,
    run_coarse_fugw,
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
    stack_slices_pairwise,
    visualize_clustered_slices,
    visualize_cluster_mapping,
    visualize_initial_connected_component,
    visualize_selected_anchors,
    visualize_global_overlap_projection,
    visualize_alignment_with_anchors,
)


# Dimensionless mesoscale constant: the neighborhood-descriptor radius is this
# multiple of the characteristic cell spacing (see ``calculate_neighborhood_dissimilarity``),
# and a supercell is sized to cover roughly one such neighborhood footprint -- so the
# coarse and fine stages of the pipeline commit to a single, consistent spatial scale.
NEIGHBORHOOD_OUTER_MULTIPLIER = 10.0


def estimate_coarsen_length(sliceA, sliceB, spatial_key="spatial"):
    """
    Derive the shared supercell seed spacing from the two slices' intrinsic geometry.

    The characteristic cell spacing ``s`` is estimated for each slice with the
    parameter-free, edge-corrected Voronoi estimator and shared as ``max(s_A, s_B)``
    (the same convention used for the neighborhood radii and the overlap grid).
    The supercell side ``S`` is then set so that one supercell covers about the
    same area as the outer neighborhood-descriptor disk of radius
    ``NEIGHBORHOOD_OUTER_MULTIPLIER * s``, i.e. ``S = sqrt(pi) * R_outer``. This
    removes the old cell-count-coupled magic number while keeping both slices at
    one shared physical scale.

    Returns:
        (S, s): the supercell seed spacing and the shared characteristic spacing.
    """
    s_A = estimate_characteristic_spacing(sliceA, spatial_key=spatial_key)
    s_B = estimate_characteristic_spacing(sliceB, spatial_key=spatial_key)
    s = max(s_A, s_B)
    r_outer = NEIGHBORHOOD_OUTER_MULTIPLIER * s
    S = float(np.sqrt(np.pi) * r_outer)
    return S, s


def _coarse_matchability(Pi_cluster, centroids_A, centroids_B):
    """
    Cross-slice matchability of a coarse coupling, for automatic scale selection.

    Specificity is the total mutual-information contribution of the coupling
    under the size-preserving independence null (higher = sharper, more
    one-to-one cluster correspondence). The rigid residual is the median
    centroid displacement of reciprocal-best cluster pairs after the best rigid
    fit, normalized by the slice-B centroid spread (lower = more geometrically
    consistent constellations). Both reuse existing hierarchical-stage utilities.
    """
    mi = float(np.sum(compute_pairwise_mutual_information_contribution(Pi_cluster)))

    if Pi_cluster.size == 0 or np.sum(Pi_cluster) <= 0:
        return mi, np.inf

    a_best = Pi_cluster.argmax(axis=1)
    b_best = Pi_cluster.argmax(axis=0)
    pairs = [
        (i, int(a_best[i]))
        for i in range(Pi_cluster.shape[0])
        if Pi_cluster[i, a_best[i]] > 0 and b_best[a_best[i]] == i
    ]
    if len(pairs) < 2:
        return mi, np.inf

    src = centroids_A[[i for i, _ in pairs]]
    tgt = centroids_B[[j for _, j in pairs]]
    R, t = fit_weighted_rigid_transform(src, tgt)
    resid = float(np.median(np.linalg.norm(tgt - (src @ R.T + t), axis=1)))
    scale = float(np.median(np.linalg.norm(centroids_B - centroids_B.mean(axis=0), axis=1))) + 1e-12
    return mi, resid / scale


def select_coarsen_length(
    sliceA,
    sliceB,
    alpha,
    delta,
    spatial_key="spatial",
    use_rep="X_pca",
    label_key="cell_type_annot",
    multipliers=(0.75, 1.0, 1.25),
    use_gpu=False,
):
    """
    Pick the supercell seed spacing that maximizes cross-slice cluster matchability.

    Candidate scales are a geometric ladder around the intrinsic default. For
    each, both slices are tessellated, the coarse partial FGW coupling is solved,
    and the pair is scored by ``_coarse_matchability``. Specificity (maximize)
    and rigid residual (minimize) are min-max normalized across candidates and
    summed; the best score wins, with the coarser scale used as a tie-break
    (coarser is cheaper downstream). This is parameter-free but ~Nx the coarse
    FGW cost, so it is opt-in.
    """
    S0, _ = estimate_coarsen_length(sliceA, sliceB, spatial_key=spatial_key)
    candidates = [float(m) * S0 for m in multipliers]

    all_types = np.array(sorted(
        set(sliceA.obs[label_key].astype(str)) | set(sliceB.obs[label_key].astype(str))
    ), dtype=str)

    _, nx = select_backend(use_gpu, gpu_verbose=False)
    scored = []  # (S, mi, resid)
    for S in candidates:
        labelsA = cluster_cells_spatial(sliceA, spatial_key=spatial_key, coarsen_length=S)
        labelsB = cluster_cells_spatial(sliceB, spatial_key=spatial_key, coarsen_length=S)
        if len(np.unique(labelsA)) < 2 or len(np.unique(labelsB)) < 2:
            continue

        cache_A = build_slice_cluster_cache(
            sliceA, labelsA, spatial_key=spatial_key, feature_key=use_rep,
            label_key=label_key, all_types=all_types,
        )
        cache_B = build_slice_cluster_cache(
            sliceB, labelsB, spatial_key=spatial_key, feature_key=use_rep,
            label_key=label_key, all_types=all_types,
        )
        M_cluster = compute_cluster_feature_costs(
            cache_A.mu_expr, cache_A.mu_struct, cache_B.mu_expr, cache_B.mu_struct, delta=delta, nx=nx,
        )
        C_A = compute_cluster_structural_matrix(cache_A.centroids)
        C_B = compute_cluster_structural_matrix(cache_B.centroids)
        Pi = run_coarse_fugw(M_cluster, C_A, C_B, cache_A.masses, cache_B.masses, alpha=alpha, use_gpu=use_gpu)

        mi, resid = _coarse_matchability(Pi, cache_A.centroids, cache_B.centroids)
        scored.append((S, mi, resid))
        print(f"  [sweep] S={S:.4g}: specificity(MI)={mi:.4g}, rigid-residual={resid:.4g}")

    if not scored:
        return S0

    S_arr = np.array([s for s, _, _ in scored], dtype=np.float64)
    mi_arr = np.array([m for _, m, _ in scored], dtype=np.float64)
    resid_arr = np.array([r for _, _, r in scored], dtype=np.float64)

    def _minmax(x, higher_is_better):
        finite = np.isfinite(x)
        y = np.where(finite, x, np.nan)
        lo, hi = np.nanmin(y), np.nanmax(y)
        if not np.isfinite(lo) or hi - lo < 1e-12:
            base = np.zeros_like(x, dtype=np.float64)
        else:
            # Non-finite entries (e.g. inf rigid residual from too few cluster pairs)
            # are filled with the worst value for that direction so they score 0,
            # not the best value that nan_to_num(nan=lo) would wrongly assign.
            nan_fill = lo if higher_is_better else hi
            base = (np.nan_to_num(y, nan=nan_fill) - lo) / (hi - lo)
        return base if higher_is_better else (1.0 - base)

    combined = _minmax(mi_arr, True) + _minmax(resid_arr, False)
    best = np.max(combined)
    winners = np.where(combined >= best - 1e-9)[0]
    return float(S_arr[winners[np.argmax(S_arr[winners])]])


def hierarchical_pairwise_align(
    sliceA: AnnData,
    sliceB: AnnData,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.25,
    alpha_cluster: float = 0.5,
    delta: float = 0.75,
    reg_m: float = 1.0,
    numItermax: int = 100000,
    use_gpu: bool = True,
    coarsen_scale: Optional[float] = None,
    auto_coarsen_scale: bool = False,
    spatial_key: str = "spatial",
    use_rep: str = "X_pca",
    label_key: str = "cell_type_annot",
    visualize_clusters: bool = False,
    verbose: bool = False,
    **kwargs
):
    """
    Performs Hierarchical OT by clustering cells into mesoregions, aligning clusters with Partial FGW,
    and then restricting the cell-level OT matchings to the aligned blocks.

    The mesoregions are uniform, contiguous supercells produced by a deterministic
    farthest-point-seeded centroidal Voronoi tessellation (see ``cluster_cells_spatial``).
    Both slices are tessellated at one shared physical scale so the supercells
    are built at one shared physical scale, giving comparable mesoregion sizes while preserving each section's observed tissue footprint -- the property the cluster-level FGW
    alignment depends on. The scale is chosen automatically; there is no
    magic-number cluster count and no per-call resolution knob.

    Args:
        use_rep: Representation used for cluster-level feature extraction in the
            coarse stage (default ``"X_pca"``). The fine-level ``pairwise_align``
            always uses raw ``slice.X`` for gene expression; this two-stage design
            is intentional — PCA gives a compact, noise-reduced signal for
            mesoregion matching, while raw counts preserve the full expression
            profile for cell-level cost computation.
        reg_m: Marginal-relaxation penalty for the unbalanced coarse FGW
            (``reg_marginals`` in POT). Higher values enforce tighter marginal
            constraints (closer to balanced transport); lower values allow more
            mass to be dropped from non-overlapping clusters. Default 1.0.
        coarsen_scale: Optional hard override of the supercell seed spacing (a
            physical length in coordinate units). When ``None`` the scale is
            derived from the slices' intrinsic geometry.
        auto_coarsen_scale: When ``True`` (and ``coarsen_scale`` is ``None``),
            select the scale by sweeping candidates and maximizing cross-slice
            cluster matchability instead of using the intrinsic default. Slower.

    Returns the cell-level alignment pi.
    """
    if verbose:
        print("--- [HOT] Step 1: Clustering Cells into Mesoregions ---")
    if coarsen_scale is not None:
        S = float(coarsen_scale)
        if verbose:
            print(f"Coarsening scale (user-supplied): S={S:.4g}")
    elif auto_coarsen_scale:
        S = select_coarsen_length(
            sliceA, sliceB,
            spatial_key=spatial_key, use_rep=use_rep, label_key=label_key,
            alpha=alpha_cluster, delta=delta, use_gpu=use_gpu,
        )
        if verbose:
            print(f"Coarsening scale (auto, matchability sweep): S={S:.4g}")
    else:
        S, s = estimate_coarsen_length(sliceA, sliceB, spatial_key=spatial_key)
        if verbose:
            print(f"Coarsening scale (intrinsic): S={S:.4g} (characteristic spacing s={s:.4g})")

    labelsA = cluster_cells_spatial(sliceA, spatial_key=spatial_key, coarsen_length=S)
    labelsB = cluster_cells_spatial(sliceB, spatial_key=spatial_key, coarsen_length=S)

    # Pre-cache global cell types for cluster structure alignment
    all_types = np.array(sorted(set(sliceA.obs[label_key].astype(str)) | set(sliceB.obs[label_key].astype(str))), dtype=str)

    if verbose:
        print(f"Slice A: {len(np.unique(labelsA))} clusters")
        print(f"Slice B: {len(np.unique(labelsB))} clusters")
    
    if visualize_clusters:
        visualize_clustered_slices(sliceA, sliceB, labelsA, labelsB, spatial_key=spatial_key)
    
    if verbose:
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
    
    if verbose:
        print("--- [HOT] Step 3: Compute Cluster Costs and Structures ---")
    _, nx_coarse = select_backend(use_gpu, gpu_verbose=False)
    M_cluster = compute_cluster_feature_costs(mu_exprA, mu_structA, mu_exprB, mu_structB, delta=delta, nx=nx_coarse)
    C_A = compute_cluster_structural_matrix(centroidsA)
    C_B = compute_cluster_structural_matrix(centroidsB)
    
    if verbose:
        print("--- [HOT] Step 4: Run Coarse FUGW ---")
    Pi_cluster = run_coarse_fugw(M_cluster, C_A, C_B, p_A, p_B, alpha=alpha_cluster, reg_m=reg_m, use_gpu=use_gpu)
    
    if visualize_clusters:
        visualize_cluster_mapping(centroidsA, centroidsB, Pi_cluster)

    # We now prepare the injection into standard cell-level pairwise_align
    if verbose:
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
        verbose=verbose,
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
    if verbose and macro_section.ambiguous:
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
    if verbose and macro_section.alternative_hypotheses:
        print(
            "[HOT] Stored competing macro-overlap hypotheses: "
            f"{len(macro_section.alternative_hypotheses)}. "
            f"Winner came from seed trial {macro_section.diagnostics.get('selected_seed_trial_rank', 1)}."
        )
    if verbose:
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

    if verbose:
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

    if verbose:
        print("--- [HOT] Step 7: Global Refinement via Overlap Projection ---")
    try:
        # 1. Geometrically align full slices using the partial block solution.
        # stack_slices_pairwise hard-codes obsm['spatial']; bridge the gap by
        # temporarily exposing the user's key under that name, then restoring it.
        _key_is_nonstandard = (spatial_key != "spatial")
        if _key_is_nonstandard:
            sliceA = sliceA.copy(); sliceA.obsm["spatial"] = sliceA.obsm[spatial_key]
            sliceB = sliceB.copy(); sliceB.obsm["spatial"] = sliceB.obsm[spatial_key]
        aligned_slices = stack_slices_pairwise([sliceA, sliceB], [pi_full], output_params=False)
        if visualize_clusters:
            visualize_alignment_with_anchors(
                aligned_slices,
                [idx_A,idx_B],
                spatial_key=spatial_key,
            )
        
        coords_A_aligned = np.asarray(aligned_slices[0].obsm["spatial"])
        coords_B_aligned = np.asarray(aligned_slices[1].obsm["spatial"])
        
        # 2. Determine overlapping regions based on Morphological Rasterization (Exact Shadow)
        # Replaces soft radius (KDTree) or rigid boundaries (Convex Hull) with a grid footprint
        # capturing the actual boundaries, contours, and internal holes perfectly.
        s_A = estimate_characteristic_spacing(sliceA, spatial_key=spatial_key)
        s_B = estimate_characteristic_spacing(sliceB, spatial_key=spatial_key)
        grid_size = max(s_A, s_B)
        
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
        # +2 guarantees the dilation (1 pixel each side) never touches the array edge.
        grid_bounds = np.maximum(max_idx_A, max_idx_B) + 2

        from scipy.ndimage import binary_dilation, generate_binary_structure
        mask_A = np.zeros(grid_bounds, dtype=bool)
        mask_B = np.zeros(grid_bounds, dtype=bool)

        mask_A[tuple(grid_A.T)] = True
        mask_B[tuple(grid_B.T)] = True

        # Cross (diamond) structuring element: expands by exactly 1 pixel along
        # the cardinal axes, avoiding the diagonal over-reach of the default
        # full-connectivity (square) element.
        cross = generate_binary_structure(2, 1)
        mask_A_dilated = binary_dilation(mask_A, structure=cross, iterations=1)
        mask_B_dilated = binary_dilation(mask_B, structure=cross, iterations=1)
        
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
        
        coords_A_orig = np.asarray(sliceA.obsm[spatial_key])
        coords_B_orig = np.asarray(sliceB.obsm[spatial_key])

        # Vectorized per-cluster spatial bandwidth (replaces per-cluster Python loop)
        cell_dists_A = np.linalg.norm(coords_A_orig - centroidsA[labelsA], axis=1)
        clust_sum_A = np.bincount(labelsA.astype(int), weights=cell_dists_A, minlength=Pi_cluster.shape[0])
        clust_cnt_A = np.bincount(labelsA.astype(int), minlength=Pi_cluster.shape[0]).astype(np.float64)
        sigma_A_clust = np.maximum(clust_sum_A / np.maximum(clust_cnt_A, 1.0), 1e-8)

        cell_dists_B = np.linalg.norm(coords_B_orig - centroidsB[labelsB], axis=1)
        clust_sum_B = np.bincount(labelsB.astype(int), weights=cell_dists_B, minlength=Pi_cluster.shape[1])
        clust_cnt_B = np.bincount(labelsB.astype(int), minlength=Pi_cluster.shape[1]).astype(np.float64)
        sigma_B_clust = np.maximum(clust_sum_B / np.maximum(clust_cnt_B, 1.0), 1e-8)

        coords_As = coords_A_orig[idx_A_shadow]   # (nA_shadow, 2)
        coords_Bs = coords_B_orig[idx_B_shadow]   # (nB_shadow, 2)

        # GPU-accelerated Gaussian soft memberships and G_init_shadow
        coords_As_t = to_backend(coords_As, nx_coarse, data_type=np.float32)
        coords_Bs_t = to_backend(coords_Bs, nx_coarse, data_type=np.float32)
        centroids_A_t = to_backend(centroidsA, nx_coarse, data_type=np.float32)
        centroids_B_t = to_backend(centroidsB, nx_coarse, data_type=np.float32)
        sigma_A_t = to_backend(sigma_A_clust, nx_coarse, data_type=np.float32)
        sigma_B_t = to_backend(sigma_B_clust, nx_coarse, data_type=np.float32)
        Pi_cluster_t = to_backend(Pi_cluster, nx_coarse, data_type=np.float32)

        diff_A = coords_As_t[:, None, :] - centroids_A_t[None, :, :]
        dA2 = nx_coarse.sum(diff_A ** 2, axis=2)
        log_SA = -0.5 * dA2 / (sigma_A_t[None, :] ** 2)
        log_SA = log_SA - nx_coarse.max(log_SA, axis=1)[:, None]
        S_A = nx_coarse.exp(log_SA)
        S_A = S_A / (nx_coarse.sum(S_A, axis=1)[:, None] + 1e-12)

        diff_B = coords_Bs_t[:, None, :] - centroids_B_t[None, :, :]
        dB2 = nx_coarse.sum(diff_B ** 2, axis=2)
        log_SB = -0.5 * dB2 / (sigma_B_t[None, :] ** 2)
        log_SB = log_SB - nx_coarse.max(log_SB, axis=1)[:, None]
        S_B = nx_coarse.exp(log_SB)
        S_B = S_B / (nx_coarse.sum(S_B, axis=1)[:, None] + 1e-12)

        G_init_shadow = nx_coarse.to_numpy(S_A @ Pi_cluster_t @ S_B.T)

        # ── Confidence-weighted marginals ─────────────────────────────────────────
        # Encode proximity to the trusted biological core as the FGW marginals rather
        # than baking it into G_init. This separates the structural prior (G_init)
        # from the confidence prior (how much mass each shadow cell contributes).
        # 95th-percentile sigma is robust to outlier shadow cells; max() is not.
        sigma_A_conf = max(float(np.percentile(dist_A[idx_A_shadow], 95)), 1e-5)
        sigma_B_conf = max(float(np.percentile(dist_B[idx_B_shadow], 95)), 1e-5)

        weight_A_shadow = np.exp(-(dist_A[idx_A_shadow] ** 2) / (2.0 * sigma_A_conf ** 2))
        weight_B_shadow = np.exp(-(dist_B[idx_B_shadow] ** 2) / (2.0 * sigma_B_conf ** 2))

        a_dist = weight_A_shadow / weight_A_shadow.sum()
        b_dist = weight_B_shadow / weight_B_shadow.sum()

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

        if verbose:
            print(f"--- [HOT] Step 8: Executing Final Base OT on Shadow Portions (A: {len(idx_A_shadow)}, B: {len(idx_B_shadow)}) ---")
        pi_shadow_final = pairwise_align(
            sliceA=sliceA_shadow,
            sliceB=sliceB_shadow,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            G_init=G_init_shadow,
            a_distribution=a_dist,
            b_distribution=b_dist,
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
        import traceback
        import warnings
        warnings.warn(
            f"[HOT] Global refinement failed and fell back to the block-restricted "
            f"coarse plan. Downstream results may be degraded.\n"
            f"Cause: {type(e).__name__}: {e}\n"
            f"{traceback.format_exc()}",
            RuntimeWarning,
            stacklevel=2,
        )
        return pi_full


def align_multiple_slices(
    slices: list[AnnData],
    spatial_key: str = "spatial",
    verbose: bool = False,
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
    
    if verbose:
        print(f"Starting Multi-Slice Alignment for {len(slices)} slices...")

    # Step 1: Compute OT matchings for all consecutive pairs
    for i in range(len(slices) - 1):
        if verbose:
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
    numItermax: int = 10000,
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
    # Computed on the active backend (GPU when available); returns a backend tensor.
    gene_cos_dist = calculate_gene_expression_cosine_distance(sliceA, sliceB, use_rep, nx=nx, data_type=data_type, eps=epsilon)


    # ────────────────────── Calculate cell-type mismatch penalty ──────────────────────
    cell_type_mismatch = to_backend(calculate_cell_type_mismatch(sliceA, sliceB), nx, data_type=data_type)


    # ────────────────────── Calculate neighborhood dissimilarity ──────────────────────
    neighborhood_jsd = to_backend(calculate_neighborhood_dissimilarity(
        sliceA,
        sliceB,
        radius=radius,                 # optional single radius; else multiscale radii are derived
        nx=nx,
        data_type=data_type,
        eps=epsilon,
        radii=None,                    # or pass an explicit list, e.g. [20, 35, 50]
        radius_k=3,
        radius_multipliers=(2.5, 4.0, 5.0),
        n_shells=3,
        harmonics=(0, 1, 2),
        distance_decay="linear",
        include_self=False,
    ), nx, data_type=data_type)

    # Combine gene expression dissimilarity, cell-type mismatch penalty and neighborhood dissimilarity into a single cost matrix M (on backend)
    M = (1 - beta - gamma) * gene_cos_dist + beta * cell_type_mismatch + gamma * neighborhood_jsd

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

    pi = fused_gromov_wasserstein_incent(M, D_A, D_B, a, b, G_init = G_init, alpha= alpha, numItermax=numItermax, verbose=verbose, **kwargs)
    pi = nx.to_numpy(pi)

    if isinstance(nx, ot.backend.TorchBackend):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pi


def _voronoi_cell_areas(coords):
    """Areas of the bounded Voronoi cells of a 2D point set.

    Cells on the convex-hull boundary have unbounded Voronoi regions and are
    skipped, so the resulting spacing estimate is edge-corrected by construction.
    Returns an empty array when a Voronoi diagram with bounded cells cannot be
    built (fewer than 4 points, (near-)collinear or duplicate input).
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[0] < 4:
        return np.empty(0, dtype=np.float64)
    try:
        vor = Voronoi(coords)
    except Exception:
        return np.empty(0, dtype=np.float64)

    areas = []
    for pt_idx in range(coords.shape[0]):
        region = vor.regions[vor.point_region[pt_idx]]
        if len(region) == 0 or -1 in region:
            continue  # unbounded region -> boundary cell -> excluded (edge correction)
        verts = vor.vertices[region]
        x, y = verts[:, 0], verts[:, 1]
        a = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))  # shoelace
        if np.isfinite(a) and a > 0:
            areas.append(a)
    return np.asarray(areas, dtype=np.float64)


def _interior_voronoi_mask(coords):
    """Boolean mask of points whose Voronoi cell is bounded (interior cells)."""
    coords = np.asarray(coords, dtype=np.float64)
    mask = np.zeros(coords.shape[0], dtype=bool)
    if coords.shape[0] < 4:
        return mask
    try:
        vor = Voronoi(coords)
    except Exception:
        return mask
    for pt_idx in range(coords.shape[0]):
        region = vor.regions[vor.point_region[pt_idx]]
        if len(region) > 0 and -1 not in region:
            mask[pt_idx] = True
    return mask


def estimate_characteristic_spacing(adata, k=3, spatial_key="spatial", method="voronoi"):
    """Robust estimate of the characteristic cell spacing of a section.

    A single length scale used to size the footprint rasterization grid, the
    cellular neighborhood radii, and the spatial-distance normalization. By
    default it is derived from the Voronoi tessellation, which needs no
    neighbor-count parameter and is edge-corrected by construction.

    Args:
        adata: Section with coordinates in ``adata.obsm[spatial_key]``.
        k: Neighbor order for the k-NN methods (ignored by ``method='voronoi'``).
            Retained for backward compatibility.
        spatial_key: Key into ``adata.obsm`` holding the (n, 2) coordinates.
        method: One of
            'voronoi'      - sqrt(median bounded-Voronoi-cell area). Parameter-free,
                             edge-corrected. Default; falls back to 'knn' if no
                             bounded cell can be formed.
            'knn_interior' - median k-th nearest-neighbor distance over interior
                             (bounded-Voronoi) cells only. Edge-corrected k-NN.
            'knn'          - median k-th nearest-neighbor distance over all cells.
                             Original behavior; kept for sensitivity analysis.

    Returns:
        The characteristic spacing as a float, or 1.0 for degenerate input.

    Notes:
        For a homogeneous 2D Poisson field of intensity lambda, the mean Voronoi
        cell area is 1/lambda, so sqrt(median area) ~ 0.96 / sqrt(lambda); the
        median k=3 NN distance is ~ 0.93 / sqrt(lambda). The two estimators agree
        to within ~4% on uniform tissue, so the radius multipliers and the
        grid factor need no retuning; the Voronoi form merely removes the
        arbitrary choice of k and the convex-hull boundary bias.
    """
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    n = coords.shape[0]
    if n < 2:
        return 1.0

    if method == "voronoi":
        areas = _voronoi_cell_areas(coords)
        if areas.size > 0:
            return float(np.sqrt(np.median(areas)))
        method = "knn"  # degenerate geometry: fall back to k-NN

    # k-NN path ('knn' and 'knn_interior'); +1 because the query returns self.
    k_eff = min(k + 1, n)
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=k_eff)
    kth = dists[:, -1]

    if method == "knn_interior":
        interior = _interior_voronoi_mask(coords)
        if interior.any():
            kth = kth[interior]

    kth = kth[np.isfinite(kth) & (kth > 0)]
    if kth.size == 0:
        return 1.0
    return float(np.median(kth))


def equal_area_shell_edges(radius, n_shells):
    """
    Equal-area shells reduce the trivial bias that outer shells cover more area.
    """
    return radius * np.sqrt(np.linspace(0.0, 1.0, n_shells + 1))


def distance_weights(dist, radius, mode="linear"):
    """
    Distance weighting for neighbors inside a radius.
    """
    if mode is None or mode == "uniform":
        return np.ones_like(dist, dtype=np.float64)

    if mode == "linear":
        return np.maximum(0.0, 1.0 - dist / radius)

    if mode == "gaussian":
        s = radius / 2.0
        return np.exp(-(dist ** 2) / (2.0 * s * s))

    raise ValueError("distance_decay must be one of: None, 'uniform', 'linear', 'gaussian'")


def neighborhood_distribution_fourier(
    adata,
    radius,
    cell_types=None,
    n_shells=3,
    harmonics=(0, 1, 2),
    distance_decay="linear",
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

    shell_edges = equal_area_shell_edges(radius, n_shells)

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

        w = distance_weights(dist, radius=radius, mode=distance_decay)
        group_idx = label_idx[nbr] * n_shells_eff + shell_idx

        local = np.zeros((n_groups, n_harm), dtype=np.float64)

        h_pos = 0
        for m in harmonics:
            if m == 0:
                mag = np.bincount(group_idx, weights=w, minlength=n_groups).astype(np.float64)
                if area_normalize:
                    mag = mag / np.maximum(group_area, 1e-12)
                local[:, h_pos] = mag
                h_pos += 1
            else:
                ang = m * theta
                real = np.bincount(group_idx, weights=w * np.cos(ang), minlength=n_groups)
                imag = np.bincount(group_idx, weights=w * np.sin(ang), minlength=n_groups)

                magnitude = np.sqrt(real ** 2 + imag ** 2)
                if area_normalize:
                    magnitude = magnitude / np.maximum(group_area, 1e-12)

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
    """Multiscale neighborhood radii as multiples of the shared characteristic spacing."""
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
    distance_decay="linear",
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

    Each radius's descriptor is built (and L1-normalized within scale) by
    :func:`neighborhood_distribution_fourier`; the per-scale blocks are concatenated
    and optionally re-normalized. Independent fine- and broad-scale views, each
    normalized in its own right, make the Jensen-Shannon neighborhood cost more
    discriminative than a single radius.
    """
    radii = [float(r) for r in radii]
    if any(r <= 0 for r in radii):
        raise ValueError("all radii must be > 0")

    blocks = []
    meta_blocks = []
    for r in radii:
        feat, meta = neighborhood_distribution_fourier(
            adata,
            radius=r,
            cell_types=cell_types,
            n_shells=n_shells,
            harmonics=harmonics,
            distance_decay=distance_decay,
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


def _pairwise_euclidean_backend(X, nx):
    """
    Pairwise Euclidean distance matrix of ``X`` (n, d) computed on the active POT
    backend. When ``nx`` is the torch backend and ``X`` lives on CUDA, the whole
    O(n^2) computation runs on the GPU and never materializes an n x n matrix on the
    host -- avoiding both the CPU compute and the host->device transfer that
    ``sklearn.euclidean_distances`` + ``to_backend`` incurred.
    """
    sq = nx.sum(X ** 2, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    D2 = nx.maximum(D2, 0.0)  # guard tiny negatives from float round-off before sqrt
    return nx.sqrt(D2)


def calculate_spatial_distance(sliceA, sliceB, nx, data_type=np.float32, spatial_key = 'spatial', eps=1e-8, norm_k=3):
    """
    Calculate spatial distance between cells in a slice, normalized by robust local spacing.

    The pairwise distances are computed directly on the active backend (GPU when
    ``nx`` is the torch backend), so no n x n matrix is built on the CPU or copied
    to the device.

    Args:
        sliceA: First slice for which to calculate spatial distance.
        sliceB: Second slice for which to calculate spatial distance.
        nx: Backend to use for calculations.
        data_type: Data type for backend tensors.
        spatial_key: Key for the spatial coordinates.
        eps: Small constant to avoid division by zero.
        norm_k: Kth neighbor to use for characteristic spacing estimation.
    Returns:
    D_A, D_B: Pairwise spatial distance matrices (backend tensors).
    """

    coordinates_A = np.asarray(sliceA.obsm[spatial_key], dtype=np.float64)
    coordinates_B = np.asarray(sliceB.obsm[spatial_key], dtype=np.float64)

    # Normalize by local characteristic spacing instead of global max tissue diameter
    scale_A = estimate_characteristic_spacing(sliceA, k=norm_k, spatial_key=spatial_key)
    scale_B = estimate_characteristic_spacing(sliceB, k=norm_k, spatial_key=spatial_key)
    scale = max(scale_A, scale_B, eps)

    # Center each slice (translation-invariant -> identical distances) before casting
    # to the backend dtype. With global coordinates far from the origin, the
    # Gram-matrix distance trick (|x|^2+|y|^2-2x.y) loses catastrophic precision in
    # float32; centering keeps magnitudes small so float32 stays accurate.
    coordinates_A = coordinates_A - coordinates_A.mean(axis=0)
    coordinates_B = coordinates_B - coordinates_B.mean(axis=0)

    coords_A = to_backend(coordinates_A, nx, data_type=data_type)
    coords_B = to_backend(coordinates_B, nx, data_type=data_type)

    D_A = _pairwise_euclidean_backend(coords_A, nx) / scale
    D_B = _pairwise_euclidean_backend(coords_B, nx) / scale

    return D_A, D_B


def calculate_gene_expression_cosine_distance(sliceA, sliceB, use_rep, nx=None, data_type=np.float32, eps = 1e-6):
    """
    Calculate cosine distance between gene expression profiles of slice A and slice B.

    Args:
    sliceA: First slice.
    sliceB: Second slice.
    use_rep: If ``None``, uses ``slice.X`` to calculate dissimilarity
                between spots, otherwise uses the representation given by ``slice.obsm[use_rep]``.
    nx: Optional POT backend. If given, the cosine distance is computed on that
        backend (GPU when it is the torch backend) and a backend tensor is
        returned. If ``None`` (default), the CPU/sklearn path is used and a numpy
        array is returned (backward compatible).
    data_type: Data type for backend tensors (when ``nx`` is given).
    eps: Small constant to add to data matrices to avoid zero vectors.

    Returns:
    cosine_dist_gene_expr: Cosine distance matrix between gene expression profiles of slice A and slice B.
    """

    # Extract and prepare data matrices for cosine distance calculation
    A_X = to_dense_array(extract_data_matrix(sliceA, use_rep)) + eps
    B_X = to_dense_array(extract_data_matrix(sliceB, use_rep)) + eps

    if nx is None:
        # CPU path: sklearn's optimized, numerically stable cosine_distances.
        return cosine_distances(A_X, B_X)

    # Backend path: normalize rows then 1 - cosine similarity, all on device.
    A = to_backend(A_X, nx, data_type=data_type)
    B = to_backend(B_X, nx, data_type=data_type)
    A = A / nx.sqrt(nx.sum(A ** 2, axis=1, keepdims=True))
    B = B / nx.sqrt(nx.sum(B ** 2, axis=1, keepdims=True))
    return 1.0 - (A @ B.T)


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
    distance_decay="linear",
    include_self=False,
    spatial_key="spatial",
    label_key="cell_type_annot",
):
    """
    Neighborhood dissimilarity from multiscale, equal-area-shell, rotation- and
    reflection-invariant descriptors, scored by Jensen-Shannon distance.

    Each cell's neighborhood is summarized, at several radii, as a distribution over
    (cell type x equal-area radial shell x angular harmonic); the per-radius blocks
    are L1-normalized within scale and concatenated. Multiple radii give independent
    fine- and broad-scale views (each normalized in its own right), which is more
    discriminative for the JS neighborhood cost than a single radius.

    Radii are chosen as (priority): the explicit ``radii`` list; else a single
    ``radius`` if given; else ``radius_multipliers`` times the shared characteristic
    cell spacing (the pipeline default). Returns the (n, m) JS-distance matrix, or
    the raw descriptors ``(featA, featB)`` when ``nx is None``.
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
                sliceA, sliceB, k=radius_k,
                multipliers=radius_multipliers, spatial_key=spatial_key,
            )

    def _descriptor(adata):
        return neighborhood_distribution_multiscale(
            adata,
            radii=radii,
            cell_types=all_types,
            n_shells=n_shells,
            harmonics=harmonics,
            distance_decay=distance_decay,
            include_self=include_self,
            area_normalize=True,
            add_empty_bin_per_scale=False,
            l1_normalize_within_scale=True,
            final_l1_normalize=True,
            dtype=np.float32,
            spatial_key=spatial_key,
            label_key=label_key,
        )

    featA = _descriptor(sliceA)
    featB = _descriptor(sliceB)

    # Empty-neighborhood cells (no neighbors within any radius) have an all-zero
    # descriptor. Assign them a uniform distribution (an uninformative prior) so the
    # Jensen-Shannon distance is defined; every other row is left exact.
    def _fill_empty_uniform(feat):
        empty = feat.sum(axis=1) <= eps
        if np.any(empty):
            feat = feat.copy()
            feat[empty] = 1.0 / feat.shape[1]
        return feat

    featA = _fill_empty_uniform(featA)
    featB = _fill_empty_uniform(featB)

    if nx is None:
        return featA, featB

    featA = to_backend(featA, nx, data_type=data_type)
    featB = to_backend(featB, nx, data_type=data_type)

    return jensenshannon_divergence_backend(featA, featB)
