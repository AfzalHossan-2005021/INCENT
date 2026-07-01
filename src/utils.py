import ot
import torch
import warnings

import numpy as np
import scipy.sparse as sp

from ot.optim import cg
from ot.gromov import init_matrix, gwloss, gwggrad, solve_gromov_linesearch


def select_backend(use_gpu=False, gpu_verbose=True):
    """
    Selects the appropriate backend (numpy or torch) based on GPU availability and user preference.

    Args:
        use_gpu: Whether to use GPU if available.
        gpu_verbose: Whether to print GPU information when selected.
    Returns:
        The selected backend module (numpy or torch).
    """
    nx = None
    if use_gpu:
        if torch.cuda.is_available():
            nx = ot.backend.TorchBackend()
            if gpu_verbose:
                print("Using gpu with Pytorch backend.")
        else:
            use_gpu = False
            nx = ot.backend.NumpyBackend()
            if gpu_verbose:
                print("CUDA is not available on your system. Reverting to CPU with Numpy backend.")
    else:
        nx = ot.backend.NumpyBackend()
        if torch.cuda.is_available() and gpu_verbose:
            print("Tip: CUDA is available on your system. You can enable GPU support by setting use_gpu=True.")
        elif gpu_verbose:
            print("Using cpu with Numpy backend.")
    return use_gpu, nx


def to_backend(x, nx, data_type=None, reference=None):
    """
    Centralized function to manage CPU-GPU movement and type consistency.
    """
    # Force to numpy safely
    if hasattr(x, 'cpu'):
        x = x.detach().cpu()
    if hasattr(x, 'numpy'):
        x = x.numpy()
    elif hasattr(x, "todense"):
        x = x.todense()
    
    # Optional typing to numpy type
    if data_type is not None:
        x = np.asarray(x, dtype=data_type)
    else:
        x = np.asarray(x)
        
    x_nx = nx.from_numpy(x)

    # Use reference tensor to match device/type if provided
    # Otherwise set up PyTorch CUDA if backend is Torch and CUDA is available
    if reference is not None: # Use POT type_as logic
        x_nx = nx.zeros(x_nx.shape, type_as=reference) + x_nx
    elif nx.__class__.__name__ == 'TorchBackend':
        import torch
        if torch.cuda.is_available():
            x_nx = x_nx.cuda()

    return x_nx


def _default_fgw_tol(x, fallback=1e-9):
    """
    Convergence tolerance matched to the working float precision.

    float32 machine epsilon is ~1.2e-7, so a 1e-9 stop threshold is unreachable and
    the solver would grind to ``numItermax``. Returns ~1e-6 for float32 (and ~1e-3
    for half precision), 1e-9 for float64. Inferred from the cost-matrix dtype.
    """
    name = str(getattr(x, "dtype", "")).lower()
    if "16" in name:      # float16 / bfloat16
        return 1e-3
    if "32" in name:      # float32
        return 1e-6
    return fallback       # float64 / double


def _project_coupling_to_marginals(G, p, q, nx):
    """
    Round a nonnegative warm start onto the transport polytope with marginals ``p``, ``q``.

    POT's conditional-gradient solver expects ``G0`` to be feasible for the same
    transport polytope used by the linear minimization oracle, but an arbitrary warm
    start (e.g. a soft lifted coupling) need not have row/column sums matching ``p``
    and ``q``. This applies the exact single-pass rounding of Altschuler, Weed &
    Rigollet (2017): scale rows down to at most ``p``, scale columns down to at most
    ``q``, then patch the leftover mass with a rank-1 correction
    ``outer(err_p, err_q) / |err_p|_1``. Because scaling only ever shrinks entries,
    both row and column sums stay below their targets until the final patch, so the
    correction term is guaranteed nonnegative and lands exactly on ``p`` and ``q`` --
    in a single O(n*m) pass, with no iteration or tolerance, and no special-casing
    needed for rows/columns that started out with zero mass.

    All arithmetic runs through ``nx`` (the active POT backend), so a CUDA ``G``/``p``/
    ``q`` stays on the GPU end to end instead of round-tripping through NumPy.
    """
    p = nx.reshape(p, (-1,))
    q = nx.reshape(q, (-1,))

    if G.shape != (p.shape[0], q.shape[0]):
        raise ValueError(
            f"G_init has shape {tuple(G.shape)}, expected {(p.shape[0], q.shape[0])} from marginals."
        )

    p = nx.maximum(nx.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    q = nx.maximum(nx.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    p_mass, q_mass = nx.sum(p), nx.sum(q)
    if p_mass <= 0 or q_mass <= 0:
        raise ValueError("Transport marginals must have positive total mass.")
    p = p / p_mass
    q = q / q_mass

    G = nx.maximum(nx.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    G_mass = nx.sum(G)
    if G_mass <= 0:
        return p[:, None] * q[None, :]
    G = G / G_mass

    # ``+ (row_sum <= 0)`` keeps the divisor away from 0 without an explicit mask op;
    # those rows are all-zero already, so the resulting scale factor is discarded.
    row_sum = nx.sum(G, axis=1)
    row_scale = nx.minimum(p / (row_sum + (1.0 - (row_sum > 0) * 1.0)), 1.0)
    G = G * row_scale[:, None]

    col_sum = nx.sum(G, axis=0)
    col_scale = nx.minimum(q / (col_sum + (1.0 - (col_sum > 0) * 1.0)), 1.0)
    G = G * col_scale[None, :]

    err_p = p - nx.sum(G, axis=1)
    err_q = q - nx.sum(G, axis=0)
    err_mass = nx.sum(err_p)
    if err_mass > 1e-14:
        G = G + nx.outer(err_p, err_q) / err_mass

    return G


def fused_gromov_wasserstein_incent(M, C1, C2, p, q, G_init = None, alpha = 0.1, log=False, numItermax=10000, numItermaxEmd=100000, tol_rel=None, tol_abs=None, verbose=False, **kwargs):
    """
    Fused Gromov-Wasserstein optimal transport with an optional warm-start coupling.

    Minimizes, over couplings ``T`` with marginals ``p`` and ``q``, a fused objective
    trading off a linear feature cost against a Gromov-Wasserstein (GW) structural
    cost::

        argmin_T  (1 - alpha) * <M, T>_F  +  alpha * GW(C1, C2, T)

    using POT's conditional-gradient solver (:func:`ot.optim.cg`). The GW term and
    its gradient are supplied as the closures ``f`` and ``df`` below, and the line
    search is either Armijo or the closed-form Gromov-Wasserstein step. Beyond a
    plain POT call, this wrapper adds: (i) an optional ``G_init`` warm-start,
    (ii) convergence tolerances that default to the working float precision, and
    (iii) execution on whatever backend/device the inputs live on (NumPy, or PyTorch
    on GPU when the inputs are CUDA tensors).

    Parameters
    ----------
    M : array-like, shape (n, m)
        Linear feature cost between source and target points (backend tensor or
        NumPy array), e.g. the combined expression/cell-type/neighborhood cost.
    C1 : array-like, shape (n, n)
        Source intra-domain structure matrix (e.g. spatial distances); symmetric.
    C2 : array-like, shape (m, m)
        Target intra-domain structure matrix; symmetric.
    p : array-like, shape (n,)
        Source marginal (mass) distribution.
    q : array-like, shape (m,)
        Target marginal (mass) distribution.
    G_init : array-like, shape (n, m), optional
        Initial coupling for the conditional gradient. It is rounded onto the
        transport polytope with marginals ``p`` and ``q`` (exact single-pass
        projection, see :func:`_project_coupling_to_marginals`) and used as the
        starting point ``G0``; ``None`` uses the outer product ``p (x) q``.
    alpha : float, default 0.1
        Trade-off in ``[0, 1]``: weight ``alpha`` on the GW structural term and
        ``1 - alpha`` on the linear feature term.
    log : bool, default False
        If ``True`` also return POT's solver log dict (which includes ``fgw_dist``).
    numItermax : int, default 10000
        Maximum number of conditional-gradient iterations.
    numItermaxEmd : int, default 100000
        Maximum iterations of the inner exact-OT (EMD) subproblem.
    tol_rel, tol_abs : float, optional
        Relative / absolute stopping thresholds. ``None`` (default) selects a value
        matched to the cost dtype (~1e-6 for float32, 1e-9 for float64), since a
        1e-9 threshold is unreachable in float32 (see :func:`_default_fgw_tol`).
    verbose : bool, default False
        Print solver progress.
    **kwargs
        Additional keyword arguments forwarded to :func:`ot.optim.cg`.

    Returns
    -------
    T : backend array, shape (n, m)
        The optimal coupling. If ``log=True``, returns ``(T, log_dict)`` instead.

    Notes
    -----
    GPU: the computation runs on the backend/device of the inputs, so passing PyTorch
    CUDA tensors (as the pipeline does when ``use_gpu=True``) keeps the whole solve
    on the GPU.
    """

    p, q = ot.utils.list_to_array(p, q)

    nx = ot.backend.get_backend(p, q, C1, C2, M)

    # Convergence thresholds default to the working float precision (1e-6 for
    # float32, 1e-9 for float64) unless explicitly provided.
    if tol_rel is None:
        tol_rel = _default_fgw_tol(M)
    if tol_abs is None:
        tol_abs = _default_fgw_tol(M)

    if G_init is None:
        G0 = p[:, None] * q[None, :]
        G0 = to_backend(G0, nx)
    else:
        # Move onto nx's backend/device first so the O(n*m) rounding below runs
        # there too (GPU-resident when nx is TorchBackend+CUDA), instead of forcing
        # it through NumPy.
        G_init = to_backend(G_init, nx, reference=M)
        G0 = _project_coupling_to_marginals(G_init, p, q, nx)

    # ── Normalize structural matrices to [0, 1] ──────────────────────────
    scale = max(nx.max(C1), nx.max(C2)) + 1e-9
    C1 = C1 / scale
    C2 = C2 / scale

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun="square_loss", nx=nx)

    def f(G):
        return gwloss(constC, hC1, hC2, G, nx=nx)

    def df(G):
        return gwggrad(constC, hC1, hC2, G, nx=nx)

    def line_search(cost, G, deltaG, Mi, cost_G, df_G, **kwargs):        
        return solve_gromov_linesearch(G, deltaG, cost_G, hC1, hC2, M=(1 - alpha) * M, reg=alpha, nx=nx, **kwargs)

    if log:
        res, log = cg(p, q, (1-alpha)*M, alpha, f, df, G0=G0, line_search=line_search, numItermax=numItermax, numItermaxEmd=numItermaxEmd, stopThr=tol_rel, stopThr2=tol_abs, verbose=verbose, log=log, nx=nx, **kwargs)
        log['fgw_dist'] = log['loss'][-1]
        return res, log

    else:
        return cg(p, q, (1-alpha)*M, alpha, f, df, G0=G0, line_search=line_search, numItermax=numItermax, numItermaxEmd=numItermaxEmd, stopThr=tol_rel, stopThr2=tol_abs, verbose=verbose, log=log, nx=nx, **kwargs)


def _is_oom_error(err) -> bool:
    """True if an exception is an out-of-memory error (CUDA or host)."""
    if isinstance(err, MemoryError):
        return True
    msg = str(err).lower()
    return "out of memory" in msg or "cuda" in msg and "memory" in msg


def _empty_cuda_cache(nx):
    if nx.__class__.__name__ == "TorchBackend":
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _elem_bytes(x, default=4):
    """Bytes per element of a numpy array or torch tensor (for memory budgeting)."""
    try:
        return int(x.dtype.itemsize)        # numpy
    except AttributeError:
        try:
            return int(x.element_size())    # torch
        except Exception:
            return default


def _auto_js_block(n, m, F, X, nx, mem_fraction):
    """
    Choose a row-block size so the (block, m, F) intermediates fit in memory.

    On the torch/CUDA backend the budget is a fraction of the *currently free*
    device memory (queried per call), scaled by the tensor's actual element size;
    otherwise a conservative fixed budget is used. The per-pair JS computation
    keeps roughly two (block, m, F) tensors live (``M`` and ``log M``), so
    ``mem_fraction`` leaves ample headroom -- and the caller additionally halves
    the block on any OOM, so an over-estimate self-corrects.
    """
    budget_elems = None
    if nx.__class__.__name__ == "TorchBackend":
        try:
            import torch
            if getattr(X, "is_cuda", False):
                free, _ = torch.cuda.mem_get_info(X.device)
                budget_elems = int(mem_fraction * free / _elem_bytes(X))
        except Exception:
            budget_elems = None
    if budget_elems is None:
        budget_elems = 30_000_000  # element budget per (block, m, F) tensor on CPU/unknown
    per_row = max(int(m) * int(F), 1)
    return int(max(1, min(int(n), budget_elems // per_row)))


def _safe_log(r, nx):
    """
    Backend ``log`` realizing the exact convention ``log(0) -> 0``.

    The zero entries it covers are always multiplied by a zero weight downstream,
    so this gives the exact ``0 * log 0 = 0`` limit with **no epsilon bias** (and
    no NaN/inf from ``0 * -inf``). For ``r > 0`` it is just ``log(r)``.
    """
    posf = (r > 0) * 1.0
    return nx.log(r + (1.0 - posf))     # r>0 -> log(r); r==0 -> log(1) = 0


def _xlogx(r, nx):
    """Elementwise ``r * log(r)`` with the exact convention ``0 * log 0 = 0``."""
    return r * _safe_log(r, nx)


def _normalize_rows(A, nx):
    """
    L1-normalize the rows of a nonnegative matrix to probability distributions.

    A zero-mass row is mapped to an all-zero row (its denominator is forced to 1)
    instead of producing ``0/0``; the returned row sums let the caller flag those
    degenerate rows afterwards.

    Returns ``(A_normalized, row_sums)``.
    """
    s = nx.sum(A, axis=1)                                 # (n,)
    denom = (s + (1.0 - (s > 0) * 1.0))[:, None]          # = s where s > 0, else 1
    return A / denom, s


def _js_distance_block(Xb, Y, xlox_b, yloy, nx):
    """
    Jensen-Shannon distance of one block of source rows against all targets.

    Parameters
    ----------
    Xb : (b, F) backend array
        Block of already-normalized source distributions.
    Y : (m, F) backend array
        All already-normalized target distributions.
    xlox_b : (b,) backend array
        Precomputed ``sum_f p log p`` for the block's rows (exact ``0 log 0 = 0``).
    yloy : (m,) backend array
        Precomputed ``sum_f q log q`` for the targets.
    nx : ot.backend.Backend
        Active POT backend.

    Returns
    -------
    (b, m) backend array
        ``sqrt(JSD(p, q))`` for each (block source ``p``, target ``q``) pair, via the
        KL form ``JSD = 1/2[(sum p log p - sum p log M) + (sum q log q - sum q log M)]``
        with midpoint ``M = (p + q) / 2``.
    """
    M = (Xb[:, None, :] + Y[None, :, :]) * 0.5            # (b, m, F)
    logM = _safe_log(M, nx)                               # log(0) -> 0 (weighted by 0)
    # einsum contracts over features without materializing the (b, m, F) products.
    sum_p_logM = nx.einsum("bf,bmf->bm", Xb, logM)
    sum_q_logM = nx.einsum("mf,bmf->bm", Y, logM)
    kl_p = xlox_b[:, None] - sum_p_logM
    kl_q = yloy[None, :] - sum_q_logM
    return nx.sqrt(nx.maximum(0.5 * (kl_p + kl_q), 0.0))


def jensenshannon_divergence_backend(X, Y, block=None, mem_fraction=0.25):
    """
    Pairwise Jensen-Shannon distance matrix between two sets of row-distributions.

    Returns ``D`` with ``D[i, j] = sqrt(JSD(X[i], Y[j]))``, the metric
    Jensen-Shannon *distance* (natural log / nats), for every pair of rows. Rows are
    L1-normalized to probability distributions first, and the computation runs on the
    active POT backend (NumPy, or PyTorch on GPU when available).

    Parameters
    ----------
    X : array-like, shape (n, F)
        Source distributions, one per row; nonnegative. NumPy array or backend tensor.
    Y : array-like, shape (m, F)
        Target distributions, one per row; nonnegative, same feature dimension as ``X``.
    block : int, optional
        Number of source rows processed per vectorized chunk. If ``None`` (default),
        it is auto-sized from free device memory (see Notes).
    mem_fraction : float, default 0.25
        Fraction of free CUDA memory the auto block size may target. Lower it if you
        observe OOM-retries; ignored on CPU.

    Returns
    -------
    backend array, shape (n, m)
        Jensen-Shannon distances. Any row/column whose input did not sum to positive
        mass is returned as ``NaN`` (its distribution is undefined).

    Raises
    ------
    ValueError
        If ``X`` and ``Y`` have different feature dimensions.

    Warns
    -----
    RuntimeWarning
        If any input row has zero total mass (its outputs are set to ``NaN``).

    Notes
    -----
    Formula. Uses the KL form ``JSD = 1/2 KL(p||M) + 1/2 KL(q||M)`` with
    ``M = (p + q) / 2`` rather than the algebraically-equal entropy identity
    ``H(M) - H(p)/2 - H(q)/2``; the latter cancels catastrophically when ``p ~ q``
    (the common near-zero-distance regime in alignment), while the KL form stays
    accurate there.

    Exactness. The convention ``0 * log 0 = 0`` is applied exactly (via
    :func:`_safe_log`), with no epsilon smoothing, so the result is unbiased and
    matches ``scipy.spatial.distance.jensenshannon`` to machine precision. Wherever
    ``p_f > 0`` the midpoint ``M_f >= p_f / 2 > 0``, so every needed log is finite;
    the only clamp covers features where both ``p`` and ``q`` vanish (weighted by 0).

    Memory. The cross term ``sum_f p log M`` cannot be factorized over pairs, so an
    exact pairwise JSD must form an ``(n, m, F)`` quantity. Source rows are therefore
    processed in blocks (only the factorizable ``sum_f p log p`` terms are precomputed
    once). The block size targets ``mem_fraction`` of free CUDA memory; on a CUDA
    out-of-memory error the block is halved and retried, so the call always completes.

    See Also
    --------
    scipy.spatial.distance.jensenshannon : single-pair reference implementation.
    """
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"X and Y must have the same number of features; got {X.shape[1]} and {Y.shape[1]}."
        )

    nx = ot.backend.get_backend(X, Y)
    n, F, m = int(X.shape[0]), int(X.shape[1]), int(Y.shape[0])

    X, sx = _normalize_rows(X, nx)
    Y, sy = _normalize_rows(Y, nx)

    # Factorizable per-row terms sum_f p log p (exact 0 log 0 = 0), computed once.
    xlox = nx.sum(_xlogx(X, nx), axis=1)                  # (n,)
    yloy = nx.sum(_xlogx(Y, nx), axis=1)                  # (m,)

    out = nx.zeros((n, m), type_as=X)
    if block is None:
        block = _auto_js_block(n, m, F, X, nx, mem_fraction)

    start = 0
    while start < n:
        stop = min(start + block, n)
        try:
            out[start:stop] = _js_distance_block(X[start:stop], Y, xlox[start:stop], yloy, nx)
            start = stop
        except (RuntimeError, MemoryError) as err:
            if _is_oom_error(err) and (stop - start) > 1:
                # the block's (b, m, F) tensors are freed when this handler exits
                # (their frame is released on `continue`); empty_cache returns that
                # memory to the allocator before the smaller retry.
                _empty_cuda_cache(nx)
                block = max(1, (stop - start) // 2)
                continue
            raise

    # Zero-mass rows/cols are undefined distributions -> NaN (scipy-consistent).
    n_degenerate = float(nx.sum(1.0 - (sx > 0) * 1.0)) + float(nx.sum(1.0 - (sy > 0) * 1.0))
    if n_degenerate > 0:
        warnings.warn(
            "jensenshannon_divergence_backend: some input rows have zero total mass "
            "(undefined distribution); their JS distances are set to NaN.",
            RuntimeWarning, stacklevel=2,
        )
        out[(sx <= 0), :] = float("nan")
        out[:, (sy <= 0)] = float("nan")

    return out


## Covert a sparse matrix into a dense np array
to_dense_array = lambda X: X.toarray() if sp.issparse(X) else np.asarray(X)

## Returns the data matrix or representation
extract_data_matrix = lambda adata,rep: adata.X if rep is None else adata.obsm[rep]

