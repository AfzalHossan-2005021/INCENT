import numpy as np
from anndata import AnnData
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon


def generalized_procrustes_analysis(
    X,
    Y,
    pi,
    output_params=False,
    matrix=False,
    allow_reflection=True,
    topk=None,
    eps=1e-12,
):
    """
    Robust post-OT rigid alignment.

    Compared with the old version, this does:
    1) coarse weighted rigid fit from the full soft pi
    2) hard partial one-to-one projection of pi using Hungarian + reject option
    3) final rigid fit on the hard support only

    Parameters
    ----------
    X : (n, 2) array
        Source coordinates.
    Y : (m, 2) array
        Target coordinates.
    pi : (n, m) array
        Soft OT coupling.
    output_params : bool
        If True, also returns transform parameters.
    matrix : bool
        If True, returns rotation/reflection matrix R.
        Else returns theta = atan2(R[1,0], R[0,0]).
        For reflected solutions, theta alone is not fully descriptive,
        so matrix=True is recommended.
    allow_reflection : bool
        Allow flips/reflections in the rigid fit.
    topk : int or None
        Number of target candidates kept per source row.
        If None, chosen automatically.
        The per-row candidate count is further clipped by the row's effective
        support so sharper transport rows keep fewer candidates.
        The retained pair is then weighted by its coupling confidence in the
        final rigid fit.
    eps : float
        Small numerical constant.

    Returns
    -------
    If output_params=False:
        X_aligned, Y_aligned

    If output_params=True and matrix=False:
        X_aligned, Y_aligned, theta, src_center, t

    If output_params=True and matrix=True:
        X_aligned, Y_aligned, R, src_center, t

    Notes
    -----
    The rigid transform is:
        x_aligned = R @ x + t
    implemented in row-vector form as:
        X_aligned = X @ R.T + t
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    pi = np.asarray(pi, dtype=float)

    assert X.ndim == 2 and Y.ndim == 2 and X.shape[1] == 2 and Y.shape[1] == 2
    assert pi.shape == (X.shape[0], Y.shape[0])

    def _auto_topk(m):
        return max(3, min(10, int(np.ceil(np.log2(max(m, 4))))))

    def _target_scale(Y_):
        if Y_.shape[0] < 2:
            return 1.0
        nn = NearestNeighbors(n_neighbors=2).fit(Y_)
        dists, _ = nn.kneighbors(Y_)
        s = float(np.median(dists[:, 1]))
        return max(s, 1.0)

    def _fit_soft_rigid(X_, Y_, W_, allow_reflection_=True):
        total = float(W_.sum())
        if total <= eps:
            raise ValueError("The transport plan mass is zero! Cannot align.")

        Wn = W_ / total
        a = Wn.sum(axis=1)
        b = Wn.sum(axis=0)

        mx = (a[:, None] * X_).sum(axis=0)
        my = (b[:, None] * Y_).sum(axis=0)

        Xc = X_ - mx
        Yc = Y_ - my

        H = Xc.T @ Wn @ Yc
        U, _, Vt = np.linalg.svd(H)
        R_ = Vt.T @ U.T

        if (not allow_reflection_) and (np.linalg.det(R_) < 0):
            Vt[-1, :] *= -1
            R_ = Vt.T @ U.T

        t_ = my - R_ @ mx
        return R_, t_, mx, my

    def _fit_pairs_rigid(Xm, Ym, w=None, allow_reflection_=True):
        if Xm.shape[0] == 0:
            raise ValueError("No matched pairs found for rigid fit.")

        if w is None:
            w = np.ones(Xm.shape[0], dtype=float)
        else:
            w = np.asarray(w, dtype=float)

        wsum = float(w.sum())
        if wsum <= eps:
            raise ValueError("Matched-pair weights sum to zero.")

        wn = w / wsum
        mx = (wn[:, None] * Xm).sum(axis=0)
        my = (wn[:, None] * Ym).sum(axis=0)

        Xc = Xm - mx
        Yc = Ym - my

        H = Xc.T @ (wn[:, None] * Yc)
        U, _, Vt = np.linalg.svd(H)
        R_ = Vt.T @ U.T

        if (not allow_reflection_) and (np.linalg.det(R_) < 0):
            Vt[-1, :] *= -1
            R_ = Vt.T @ U.T

        t_ = my - R_ @ mx
        return R_, t_, mx, my

    # ------------------------------------------------------------------
    # 1) Coarse soft rigid fit from full pi
    # ------------------------------------------------------------------
    pi_mass = float(pi.sum())
    if pi_mass <= eps:
        raise ValueError("The transport plan mass is zero! Cannot align.")

    R0, t0, src_center0, _ = _fit_soft_rigid(X, Y, pi, allow_reflection_=allow_reflection)

    # ------------------------------------------------------------------
    # 2) Hard partial one-to-one projection of pi
    # ------------------------------------------------------------------
    n, m = pi.shape
    row_mass = pi.sum(axis=1)
    active_rows = np.where(row_mass > eps)[0]

    if len(active_rows) < 2:
        # Too little support to do anything more robust than the soft fit
        X_aligned = X @ R0.T + t0
        Y_aligned = Y.copy()

        if not output_params:
            return X_aligned, Y_aligned

        theta = float(np.arctan2(R0[1, 0], R0[0, 0]))
        if matrix:
            return X_aligned, Y_aligned, R0, src_center0, t0
        return X_aligned, Y_aligned, theta, src_center0, t0

    s = _target_scale(Y)
    topk = _auto_topk(m) if topk is None else max(1, min(int(topk), m))

    all_real_cols = set()
    cand_cols_per_row = []
    real_costs_per_row = []
    row_cond_per_row = []

    for i in active_rows:
        cond = pi[i] / max(row_mass[i], eps)

        row_support = 1.0 / max(float(np.sum(cond ** 2)), eps)
        row_topk = max(1, min(topk, int(np.ceil(row_support))))

        cand = np.argsort(-cond)[:row_topk]
        cand = cand[cond[cand] > eps]

        if cand.size == 0:
            cand = np.array([int(np.argmax(pi[i]))], dtype=int)

        x_pred = X[i] @ R0.T + t0
        resid = np.linalg.norm(Y[cand] - x_pred[None, :], axis=1) / s
        prob_cost = -np.log(cond[cand] + eps)

        # Both terms are dimensionless:
        # -log(prob) rewards confident OT mass
        # residual/s rewards rigid consistency with the coarse transform
        cost = prob_cost + resid

        cand_cols_per_row.append(cand)
        real_costs_per_row.append(cost)
        row_cond_per_row.append(cond)
        all_real_cols.update(cand.tolist())

    all_real_cols = np.array(sorted(all_real_cols), dtype=int)
    real_col_to_idx = {j: idx for idx, j in enumerate(all_real_cols)}

    nr = len(active_rows)
    nc_real = len(all_real_cols)
    BIG = 1e9
    C = np.full((nr, nc_real + nr), BIG, dtype=float)

    for r, (i, cand, cost) in enumerate(zip(active_rows, cand_cols_per_row, real_costs_per_row)):
        for j, c in zip(cand, cost):
            C[r, real_col_to_idx[int(j)]] = float(c)

        # Private dummy for this row -> allows unmatched source rows
        med = float(np.median(cost))
        mad = float(np.median(np.abs(cost - med)))
        reject_cost = med + 1.4826 * mad
        if not np.isfinite(reject_cost):
            reject_cost = med + 1.0
        if reject_cost <= float(np.min(cost)):
            reject_cost = float(np.min(cost) + max(0.5, mad + 0.5))

        C[r, nc_real + r] = reject_cost

    row_ind, col_ind = linear_sum_assignment(C)

    pairs = []
    weights = []

    for r, c in zip(row_ind, col_ind):
        if c >= nc_real:
            continue  # matched to private dummy -> leave row unmatched

        i = int(active_rows[r])
        j = int(all_real_cols[c])
        pairs.append((i, j))
        # Confidence-weight the retained match so the strongest pairs dominate.
        weights.append(row_mass[i] * float(row_cond_per_row[r][j]))

    # ------------------------------------------------------------------
    # 3) Final rigid fit on hard support only
    # ------------------------------------------------------------------
    if len(pairs) >= 2:
        src_idx = np.array([i for i, _ in pairs], dtype=int)
        tgt_idx = np.array([j for _, j in pairs], dtype=int)
        w = np.asarray(weights, dtype=float)

        R, t, src_center, _ = _fit_pairs_rigid(
            X[src_idx],
            Y[tgt_idx],
            w=w,
            allow_reflection_=allow_reflection,
        )
    else:
        R, t, src_center = R0, t0, src_center0

    X_aligned = X @ R.T + t
    Y_aligned = Y.copy()

    if not output_params:
        return X_aligned, Y_aligned

    theta = float(np.arctan2(R[1, 0], R[0, 0]))
    if matrix:
        return X_aligned, Y_aligned, R, src_center, t
    return X_aligned, Y_aligned, theta, src_center, t


def stack_slices_pairwise(
    slices: List[AnnData],
    pis: List[np.ndarray],
    output_params: bool = False,
    matrix: bool = False
) -> Tuple[List[AnnData], Optional[List[float]], Optional[List[np.ndarray]]]:
    """
    Align spatial coordinates of sequential pairwise slices.

    This function anchors all slices to the coordinate system of slices[0].
    Transformations are accumulated through the stack sequentially:
    slices[1] -> slices[0], then slices[2] -> slices[1] (now in 0's space), etc.
    """
    assert len(slices) == len(pis) + 1, "'slices' should have length one more than 'pis'. Please double check."
    assert len(slices) > 1, "You should have at least 2 layers."
    
    new_coor = [slices[0].obsm['spatial'].copy()]
    thetas = []
    translations = []
    
    # We iterate forward, mapping slice i+1 to slice i's newly aligned coordinates
    for i in range(len(slices) - 1):
        # pis[i] maps slices[i] -> slices[i+1]
        # To align i+1 to i, X = slices[i+1], Y = new_coor[i] (which is slices[i] in the global space), pi is transposed!
        if not output_params:
            X_aligned, _ = generalized_procrustes_analysis(
                slices[i+1].obsm['spatial'], 
                new_coor[i], 
                pis[i].T
            )
            new_coor.append(X_aligned)
        else:
            X_aligned, _, theta, src_center, t = generalized_procrustes_analysis(
                slices[i+1].obsm['spatial'], 
                new_coor[i], 
                pis[i].T,
                output_params=output_params, 
                matrix=matrix
            )
            new_coor.append(X_aligned)
            thetas.append(theta)
            translations.append(t) # Only append the translation vector, ignoring src_center to fix output struct

    new_slices = []
    for i in range(len(slices)):
        s = slices[i].copy()
        s.obsm['spatial'] = new_coor[i]
        new_slices.append(s)

    if not output_params:
        return new_slices
    else:
        return new_slices, thetas, translations


def visualize_alignment(
    slices: List[AnnData],
    pis: List[np.ndarray],
    spatial_key: str = "spatial",
    alpha: float = 0.5,
    s: float = 1.0,
    colors: list = None
):
    """
    Visualizes the 2D alignment of multiple sequential slices after Procrustes refinement.
    """
    new_slices = stack_slices_pairwise(slices, pis)
    n_slices = len(new_slices)

    if colors is None:
        if n_slices == 2:
            colors = ['#e41a1c', '#377eb8']  # Red (Source/0), Blue (Target/1)
        else:
            cmap = plt.get_cmap('tab20')
            colors = [cmap(i % 20) for i in range(n_slices)]
    elif len(colors) < n_slices:
        colors = [colors[i % len(colors)] for i in range(n_slices)]

    print(f"====================\nAligned {n_slices} slices")
    fig = plt.figure(figsize=(8, 8))

    for idx, (slice_obj, color) in enumerate(zip(new_slices, colors)):
        coords = slice_obj.obsm[spatial_key]
        label = f'Slice {idx} (Anchor)' if idx == 0 else f'Slice {idx}'
        plt.scatter(coords[:, 0], coords[:, 1], s=s, alpha=alpha, label=label, c=[color]*len(coords))

    plt.axis("equal")
    plt.axis("off")
    if n_slices <= 10:
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    plt.tight_layout()
    plt.show()

    return new_slices


def visualize_alignment_with_anchors(
    slices: List[AnnData],
    anchor_indices: List[np.ndarray],
    spatial_key: str = "spatial",
    alpha: float = 0.5,
    s: float = 1.0,
    colors: list = None
):
    """
    Visualizes the 2D alignment of multiple sequential slices after Procrustes refinement.
    Cells outside the selected anchors are shown in gray; anchor cells retain their slice color.

    Args:
        slices: List of AnnData objects to align and visualize.
        pis: List of OT transport matrices (len = len(slices) - 1).
        anchor_indices: List of index arrays (one per slice) marking the anchor/overlap cells.
                        Pass None for a slice to color all its cells normally.
        spatial_key: Key in .obsm storing spatial coordinates.
        alpha: Transparency for anchor (colored) points.
        s: Point size.
        colors: Optional per-slice colors. Defaults to red/blue for 2 slices, tab20 otherwise.
    """
    n_slices = len(slices)

    if colors is None:
        if n_slices == 2:
            colors = ['#e41a1c', '#377eb8']
        else:
            cmap = plt.get_cmap('tab20')
            colors = [cmap(i % 20) for i in range(n_slices)]
    elif len(colors) < n_slices:
        colors = [colors[i % len(colors)] for i in range(n_slices)]

    print(f"====================\nAligned {n_slices} slices")
    fig = plt.figure(figsize=(8, 8))

    for idx, (slice_obj, color) in enumerate(zip(slices, colors)):
        coords = slice_obj.obsm[spatial_key]
        n_cells = coords.shape[0]
        label = f'Slice {idx} (Anchor)' if idx == 0 else f'Slice {idx}'

        anchor_idx = anchor_indices[idx] if idx < len(anchor_indices) else None

        if anchor_idx is None or len(anchor_idx) == n_cells:
            # All cells are anchors — same as visualize_alignment
            plt.scatter(coords[:, 0], coords[:, 1], s=s, alpha=alpha, label=label, c=[color] * n_cells)
        else:
            anchor_mask = np.zeros(n_cells, dtype=bool)
            anchor_mask[anchor_idx] = True

            # Gray for non-anchor cells (no label to keep legend clean)
            if not anchor_mask.all():
                plt.scatter(
                    coords[~anchor_mask, 0], coords[~anchor_mask, 1],
                    s=s, alpha=alpha, c='lightgray', zorder=1
                )

            # Colored for anchor cells
            if anchor_mask.any():
                plt.scatter(
                    coords[anchor_mask, 0], coords[anchor_mask, 1],
                    s=s, alpha=alpha, label=label, c=[color] * int(anchor_mask.sum()), zorder=2
                )

    plt.axis("equal")
    plt.axis("off")
    if n_slices <= 10:
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    plt.tight_layout()
    plt.show()


def visualize_3d_stack(
    slices: list,
    pi_matrices: list,
    spatial_key: str = "spatial",
    z_spacing: float = 10.0,
    point_size: float = 2.0,
    alpha: float = 0.5,
    colors: list = None
):
    """
    Renders a 3D scatter plot of sequentially aligned spatial transcriptomics slices.
    
    Args:
        aligned_ slices: List of AnnData objects already aligned to a common coordinate system.
        spatial_key: Key in `.obsm` storing the spatial coordinates.
        z_spacing: Artificial Z-gap distance placed between each sequential slice.
        point_size: Global scatter plot point size.
        alpha: Transparency of the points.
        colors: Optional list of hex/named colors for each slice. If None, uses a colorful colormap.
    """
    import matplotlib.pyplot as plt
    try:
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("To plot in 3D, mpl_toolkits.mplot3d must be installed.")
        return
    
    # Step 2: Global geometric assembly using the Procrustes chain
    print("\n--- Assembling Global Coordinate Stack ---")
    aligned_slices = stack_slices_pairwise(slices, pi_matrices, output_params=False)

    n_slices = len(aligned_slices)
    if n_slices < 1:
        print("No slices to visualize in 3D.")
        return

    # Auto-generate colors if not provided
    if colors is None:
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i % 20) for i in range(n_slices)]
    elif len(colors) < n_slices:
        print(f"Warning: Only {len(colors)} colors provided for {n_slices} slices. Repeating colors.")
        colors = [colors[i % len(colors)] for i in range(n_slices)]

    fig = plt.figure(figsize=(10, 8))
    # 'add_subplot' with projection='3d' automatically invokes the imported Axes3D
    ax = fig.add_subplot(111, projection='3d')

    for idx, (s, c) in enumerate(zip(aligned_slices, colors)):
        coords = s.obsm[spatial_key]
        x = coords[:, 0]
        y = coords[:, 1]
        # Generate artificial Z coordinate spacing
        z = np.full(x.shape, float(idx * z_spacing))

        ax.scatter(x, y, z, c=[c]*len(x), s=point_size, alpha=alpha, label=f"Slice {idx}")

    ax.set_xlabel('Global X')
    ax.set_ylabel('Global Y')
    ax.set_zlabel(f'Z-Stack (Spacing={z_spacing})')
    ax.set_title(f'3D Reconstruction of {n_slices} Aligned Slices')

    # Improve view angles
    ax.view_init(elev=20., azim=45)
    
    # Legend max limited to prevent huge unreadable labels inside plot
    if n_slices <= 10:
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
    plt.tight_layout()
    plt.show()


def visualize_clustered_slices(sliceA, sliceB, labelsA, labelsB, spatial_key="spatial"):
    """
    Visualizes the spatial clusters of two aligned slices side by side.
    
    Args:
        sliceA: First AnnData slice (already aligned).
        sliceB: Second AnnData slice (already aligned).
        labelsA: Cluster labels for sliceA (array-like).
        labelsB: Cluster labels for sliceB (array-like).
        spatial_key: Key in .obsm where spatial coordinates are stored.
    """
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ptsA = sliceA.obsm[spatial_key]
    ptsB = sliceB.obsm[spatial_key]
    # using categorical cmap
    cmap = plt.get_cmap('tab20')
    ax1.scatter(ptsA[:,0], ptsA[:,1], c=labelsA, cmap=cmap, s=2, alpha=0.8)
    ax1.set_title(f"Slice A: {len(np.unique(labelsA))} Clusters")
    ax1.axis('equal')
    ax2.scatter(ptsB[:,0], ptsB[:,1], c=labelsB, cmap=cmap, s=2, alpha=0.8)
    ax2.set_title(f"Slice B: {len(np.unique(labelsB))} Clusters")
    ax2.axis('equal')
    plt.show()


def visualize_cluster_mapping(
    centroidsA,
    centroidsB,
    Pi_cluster,
    block_threshold=1e-4
):
    """
    Visualizes the macro-level cluster matching between two slices using centroids and the Pi_cluster matrix.
    Args:
        centroidsA: (C_A, 2) array of cluster centroids for slice A.
        centroidsB: (C_B, 2) array of cluster centroids for slice B.
        Pi_cluster: (C_A, C_B) array of cluster-level transport plan values.
        block_threshold: Minimum Pi_cluster value to draw a connection line between clusters.
    """
    # Center the coordinates purely for overlap plotting
    cA_plot = centroidsA - np.mean(centroidsA, axis=0)
    cB_plot = centroidsB - np.mean(centroidsB, axis=0)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(cA_plot[:,0], cA_plot[:,1], c='blue', s=20, label='Slice A Clusters (Centered)', zorder=2)
    ax.scatter(cB_plot[:,0], cB_plot[:,1], c='red', s=20, label='Slice B Clusters (Centered)', zorder=2)

    max_pi = np.max(Pi_cluster)
    if max_pi > 0:
        lines = []
        linewidths = []
        for i in range(Pi_cluster.shape[0]):
            for j in range(Pi_cluster.shape[1]):
                if Pi_cluster[i, j] > block_threshold:
                    lines.append([(cA_plot[i,0], cA_plot[i,1]), (cB_plot[j,0], cB_plot[j,1])])
                    linewidths.append((Pi_cluster[i, j] / max_pi) * 2.0)      

        lc = LineCollection(lines, colors='k', linewidths=linewidths, alpha=0.5, zorder=1)
        ax.add_collection(lc)

    ax.set_title("Macro-Level Cluster Matching ($Pi_{cluster}$)")
    ax.axis('equal')
    ax.legend()
    plt.show()


def visualize_selected_anchors(
    sliceA,
    sliceB,
    idx_A,
    idx_B,
    dist_A,
    dist_B,
    spatial_key="spatial"
):
    """
    Visualizes the selected anchor clusters and their distance gradients in two aligned slices.
    Args:
        sliceA: First AnnData slice (already aligned).
        sliceB: Second AnnData slice (already aligned).
        idx_A: Index of the selected core in sliceA.
        idx_B: Index of the selected core in sliceB.
        dist_A: Array of distances from the selected core for all points in sliceA.
        dist_B: Array of distances from the selected core for all points in sliceB.
        spatial_key: Key in .obsm where spatial coordinates are stored.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ptsA = sliceA.obsm[spatial_key]
    ptsB = sliceB.obsm[spatial_key]
    
    # Highlight chosen core dynamically
    core_A_mask = np.zeros(sliceA.shape[0], dtype=bool)
    core_A_mask[idx_A] = True
    core_B_mask = np.zeros(sliceB.shape[0], dtype=bool)
    core_B_mask[idx_B] = True
    
    # Plot distance out from core (distance is 0 inside core)
    sc1 = ax1.scatter(ptsA[:,0], ptsA[:,1], c=dist_A, cmap='viridis_r', s=4, alpha=0.9)
    ax1.scatter(ptsA[core_A_mask,0], ptsA[core_A_mask,1], c='red', s=2, alpha=0.5, label='Selected Core')
    ax1.set_title("Slice A: Macro Selection & Distances")
    ax1.axis('equal')
    ax1.legend()
    fig.colorbar(sc1, ax=ax1, label='Distance to Core', fraction=0.046, pad=0.04)
    
    sc2 = ax2.scatter(ptsB[:,0], ptsB[:,1], c=dist_B, cmap='viridis_r', s=4, alpha=0.9)
    ax2.scatter(ptsB[core_B_mask,0], ptsB[core_B_mask,1], c='red', s=2, alpha=0.5, label='Selected Core')
    ax2.set_title("Slice B: Macro Selection & Distances")
    ax2.axis('equal')
    ax2.legend()
    fig.colorbar(sc2, ax=ax2, label='Distance to Core', fraction=0.046, pad=0.04)
    
    plt.show()


def visualize_initial_connected_component(
    sliceA,
    sliceB,
    idx_A,
    idx_B,
    spatial_key="spatial"
):
    """
    Visualizes the initially selected connected macro component before later expansion.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ptsA = sliceA.obsm[spatial_key]
    ptsB = sliceB.obsm[spatial_key]

    init_A_mask = np.zeros(sliceA.shape[0], dtype=bool)
    init_A_mask[idx_A] = True
    init_B_mask = np.zeros(sliceB.shape[0], dtype=bool)
    init_B_mask[idx_B] = True

    ax1.scatter(ptsA[:, 0], ptsA[:, 1], c='lightgray', s=4, alpha=0.35, label='All Cells')
    ax1.scatter(ptsA[init_A_mask, 0], ptsA[init_A_mask, 1], c='crimson', s=5, alpha=0.9, label='Initial Connected Component')
    ax1.set_title("Slice A: Initial Connected Component")
    ax1.axis('equal')
    ax1.legend()

    ax2.scatter(ptsB[:, 0], ptsB[:, 1], c='lightgray', s=4, alpha=0.35, label='All Cells')
    ax2.scatter(ptsB[init_B_mask, 0], ptsB[init_B_mask, 1], c='crimson', s=5, alpha=0.9, label='Initial Connected Component')
    ax2.set_title("Slice B: Initial Connected Component")
    ax2.axis('equal')
    ax2.legend()

    plt.show()


def visualize_global_overlap_projection(
    sliceA,
    sliceB,
    idx_A_shadow,
    idx_B_shadow,
    weight_A_shadow,
    weight_B_shadow,
    spatial_key="spatial"
):
    """
    Visualizes the global overlap projection weights across the entire slices, showing how the shadow portion is distributed spatially with the core decay.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ptsA_full = sliceA.obsm[spatial_key]
    ptsB_full = sliceB.obsm[spatial_key]
    
    weight_A_full = np.zeros(sliceA.shape[0])
    weight_A_full[idx_A_shadow] = weight_A_shadow
    
    weight_B_full = np.zeros(sliceB.shape[0])
    weight_B_full[idx_B_shadow] = weight_B_shadow
    
    sc1 = ax1.scatter(ptsA_full[:,0], ptsA_full[:,1], c=weight_A_full, cmap='magma', s=2, alpha=0.9)
    ax1.set_title("Slice A: Shadow Weights (Core Decaying)")
    ax1.axis('equal')
    fig.colorbar(sc1, ax=ax1, label='Init Weight Bias')
    
    sc2 = ax2.scatter(ptsB_full[:,0], ptsB_full[:,1], c=weight_B_full, cmap='magma', s=2, alpha=0.9)
    ax2.set_title("Slice B: Shadow Weights (Core Decaying)")
    ax2.axis('equal')
    fig.colorbar(sc2, ax=ax2, label='Init Weight Bias')
    
    plt.suptitle("Global Overlap Projection Weights (Exact Shadow + Core Decay)")
    plt.show()
