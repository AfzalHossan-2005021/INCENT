import ot
import torch

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from ot.optim import line_search_armijo, cg
from ot.gromov import solve_gromov_linesearch


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


def fused_gromov_wasserstein_incent(M, C1, C2, p, q, G_init = None, alpha = 0.1, armijo=False, log=False, numItermax=6000, numItermaxEmd=100000, tol_rel=None, tol_abs=None, verbose=False, **kwargs):
    """
    This method is written by Anup Bhowmik, CSE, BUET

    Adapted fused_gromov_wasserstein with the added capability of defining a G_init (inital mapping).
    Also added capability of utilizing different POT backends to speed up computation.
    
    For more info, see: https://pythonot.github.io/gen_modules/ot.gromov.html

    # M: combined cost matrix (M1 + gamma * M2)
    # C1: spatial distance matrix of slice 1
    # C2: spatial distance matrix of slice 2

    # p: initial distribution(uniform) of sliceA spots
    # q: initial distribution(uniform) of sliceB spots

    # how did they incorporate the spatial data in the fused gromov wasserstein?
    # C1: spatial distance matrix of slice 1
    # C2: spatial distance matrix of slice 2
    # p: gene expression distribution of slice 1 (initial distribution is uniform)
    # q: gene expression distribution of slice 2
    # G_init: initial pi matrix mapping
    # loss_fun: loss function to use (square loss)
    # alpha: step size
    # armijo: whether to use armijo line search
    # log: whether to print log
    # numItermax: maximum number of iterations
    # tol_rel: relative tolerance
    # tol_abs: absolute tolerance
    # use_gpu: whether to use gpu
    # **kwargs: additional arguments for ot.gromov.fgw

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
    else:
        G0 = (1/nx.sum(G_init)) * G_init
    G0 = to_backend(G0, nx)

    def f(G):
        # Base Gromov-Wasserstein term
        return nx.sum((G @ G.T)  * C1) + nx.sum((G.T @ G)  * C2)

    def df(G):
        # Gradient of GW term
        return 2 * (nx.dot(C1, G) + nx.dot(G, C2))

    if armijo:
        def line_search(cost, G, deltaG, Mi, cost_G, df_G, **kwargs):
            alpha_step, fc, cost_new = line_search_armijo(cost, G, deltaG, Mi, cost_G, nx=nx, **kwargs)
            # Enforce probability simplex limit to avoid negative masses entirely
            if alpha_step is None: alpha_step = 1.0
            if alpha_step > 1.0: alpha_step = 1.0
            if alpha_step < 0.0: alpha_step = 0.0
            cost_new = cost(G + alpha_step * deltaG)
            return alpha_step, fc, cost_new
    else:
        def line_search(cost, G, deltaG, Mi, cost_G, df_G, **kwargs):
            # enforce hard bounds natively from solve_1d_linesearch_quad
            return solve_gromov_linesearch(G, deltaG, cost_G, C1, C2, M=(1-alpha)*M, reg=alpha, alpha_min=0.0, alpha_max=1.0, nx=nx, **kwargs)

    if log:
   
        res, log = cg(p, q, (1-alpha)*M, alpha, f, df, G0=G0, line_search=line_search, numItermax=numItermax, numItermaxEmd=numItermaxEmd, stopThr=tol_rel, stopThr2=tol_abs, verbose=verbose, log=log, nx=nx, **kwargs)

        fgw_dist = log['loss'][-1]

        log['fgw_dist'] = fgw_dist
        log['u'] = log['u']
        log['v'] = log['v']
        return res, log

    else:
        return cg(p, q, (1-alpha)*M, alpha, f, df, G0=G0, line_search=line_search, numItermax=numItermax, numItermaxEmd=numItermaxEmd, stopThr=tol_rel, stopThr2=tol_abs, verbose=verbose, log=log, nx=nx, **kwargs)


def kl_divergence_corresponding_backend(X, Y):
    """
    Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.

    Takes advantage of POT backend to speed up computation.

    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)

    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    nx = ot.backend.get_backend(X,Y)

    epsilon = 1e-12

    X = X/(nx.sum(X,axis=1, keepdims=True) + epsilon)
    Y = Y/(nx.sum(Y,axis=1, keepdims=True) + epsilon)
    log_X = nx.log(X + epsilon)
    log_Y = nx.log(Y + epsilon)
    X_log_X = nx.einsum('ij,ij->i',X,log_X)
    X_log_X = nx.reshape(X_log_X,(1,X_log_X.shape[0]))

    X_log_Y = nx.einsum('ij,ij->i',X,log_Y)
    X_log_Y = nx.reshape(X_log_Y,(1,X_log_Y.shape[0]))
    D = X_log_X.T - X_log_Y.T
    return D


def jensenshannon_distance_1_vs_many_backend(X, Y):
    """
    Returns pairwise Jensenshannon distance (over all pairs of samples) of two matrices X and Y.

    Takes advantage of POT backend to speed up computation.

    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)

    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."
    assert X.shape[0] == 1
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nx = ot.backend.get_backend(X,Y)        # np or torch depending upon gpu availability
    X = nx.concatenate([X] * Y.shape[0], axis=0) # broadcast X
    X = X/(nx.sum(X,axis=1, keepdims=True) + 1e-12)   # normalize
    Y = Y/(nx.sum(Y,axis=1, keepdims=True) + 1e-12)   # normalize
    M = (X + Y) / 2.0
    kl_X_M = kl_divergence_corresponding_backend(X, M)
    kl_Y_M = kl_divergence_corresponding_backend(Y, M)
    # Clip small negative values due to floating point error before sqrt
    js_sq = (kl_X_M + kl_Y_M) / 2.0
    js_dist = nx.sqrt(nx.maximum(js_sq, 0.0)).T[0]
    return js_dist


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


def _auto_js_block(n, m, F, X, nx, mem_fraction, dtype_bytes=4):
    """
    Choose a row-block size so the (block, m, F) intermediates fit in memory.

    On the torch/CUDA backend the budget is a fraction of the *currently free*
    device memory (queried per call); otherwise a conservative fixed budget is
    used. The per-pair JS computation keeps roughly two (block, m, F) tensors
    live (``M`` and ``log M``), so ``mem_fraction`` leaves ample headroom -- and
    the caller additionally halves the block on any OOM, so an over-estimate
    self-corrects.
    """
    budget_elems = None
    if nx.__class__.__name__ == "TorchBackend":
        try:
            import torch
            if getattr(X, "is_cuda", False):
                free, _ = torch.cuda.mem_get_info(X.device)
                budget_elems = int(mem_fraction * free / dtype_bytes)
        except Exception:
            budget_elems = None
    if budget_elems is None:
        budget_elems = 30_000_000  # ~120 MB per (block, m, F) tensor on CPU/unknown
    per_row = max(int(m) * int(F), 1)
    return int(max(1, min(int(n), budget_elems // per_row)))


def jensenshannon_divergence_backend(X, Y, block=None, mem_fraction=0.25, eps=1e-12):
    """
    Pairwise Jensen-Shannon distance matrix (n x m) of row-distributions X, Y.

    Computed in row-blocks that are each fully vectorized on the active POT backend
    (GPU when available) -- this removes the per-row Python loop while bounding the
    peak ``(block, m, F)`` intermediate so CUDA does not run out of memory. The
    block size is derived from free device memory (see :func:`_auto_js_block`); on a
    CUDA OOM the block is halved and retried, so the computation always completes.

    Exactness note: JSD needs the entropy of the midpoint ``M = (p + q) / 2``, whose
    ``sum_f M log M`` term does not factorize across pairs, so an exact pairwise JSD
    must form an ``(n, m, F)`` quantity -- hence chunking rather than a single
    vectorized expression. The per-row ``sum_f p log p`` terms DO factorize and are
    precomputed once.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    nx = ot.backend.get_backend(X, Y)

    X = X / (nx.sum(X, axis=1, keepdims=True) + eps)
    Y = Y / (nx.sum(Y, axis=1, keepdims=True) + eps)

    n, F = int(X.shape[0]), int(X.shape[1])
    m = int(Y.shape[0])

    out = nx.zeros((n, m), type_as=X)
    # Factorizable per-row entropies, computed once.
    xlox = nx.sum(X * nx.log(X + eps), axis=1)          # (n,)
    yloy = nx.sum(Y * nx.log(Y + eps), axis=1)          # (m,)

    if block is None:
        block = _auto_js_block(n, m, F, X, nx, mem_fraction)

    start = 0
    while start < n:
        stop = min(start + block, n)
        try:
            Xb = X[start:stop]                                  # (b, F)
            M = (Xb[:, None, :] + Y[None, :, :]) * 0.5          # (b, m, F)
            logM = nx.log(M + eps)                              # (b, m, F)
            # KL(p||M) = sum_f p log p - sum_f p log M ; same for q. einsum avoids
            # materializing the (b, m, F) elementwise products.
            sum_x_logM = nx.einsum("bf,bmf->bm", Xb, logM)
            sum_y_logM = nx.einsum("mf,bmf->bm", Y, logM)
            kl_x = xlox[start:stop][:, None] - sum_x_logM
            kl_y = yloy[None, :] - sum_y_logM
            out[start:stop] = nx.sqrt(nx.maximum(0.5 * (kl_x + kl_y), 0.0))
            del M, logM, sum_x_logM, sum_y_logM, kl_x, kl_y
            start = stop
        except (RuntimeError, MemoryError) as err:
            if _is_oom_error(err) and (stop - start) > 1:
                _empty_cuda_cache(nx)
                block = max(1, (stop - start) // 2)
                continue
            raise

    return out


## Covert a sparse matrix into a dense np array
to_dense_array = lambda X: X.toarray() if sp.issparse(X) else np.asarray(X)

## Returns the data matrix or representation
extract_data_matrix = lambda adata,rep: adata.X if rep is None else adata.obsm[rep]

