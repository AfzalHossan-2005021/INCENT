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


def fused_gromov_wasserstein_incent(M, C1, C2, p, q, G_init = None, alpha = 0.1, armijo=False, log=False, numItermax=6000, numItermaxEmd=100000, tol_rel=1e-9, tol_abs=1e-9, verbose=False, **kwargs):
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


def jensenshannon_divergence_backend(X, Y):
    """    
    Returns pairwise JS divergence (over all pairs of samples) of two matrices X and Y.

    Takes advantage of POT backend to speed up computation.

    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)

    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    print("Calculating cost matrix")

    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    nx = ot.backend.get_backend(X,Y)        
    
    X = X/(nx.sum(X,axis=1, keepdims=True) + 1e-12)
    Y = Y/(nx.sum(Y,axis=1, keepdims=True) + 1e-12)

    n = X.shape[0]
    m = Y.shape[0]
    
    js_dist = nx.zeros((n, m))

    for i in tqdm(range(n)):
        js_dist[i, :] = jensenshannon_distance_1_vs_many_backend(X[i:i+1], Y)
        
    print("Finished calculating cost matrix")

    return js_dist

def pairwise_msd(A, B):
    """
    Returns pairwise mean squared distance (over all pairs of samples) of two matrices A and B.

    Args:
        A: np array with dim (m_samples by d_features)
        B: np array with dim (n_samples by d_features)
    """
    A = np.asarray(A)
    B = np.asarray(B)

    # A: (m, d), B: (n, d)
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]  # shape: (m, n, d)
    msd = np.mean(diff ** 2, axis=2)  # shape: (m, n)
    return msd

## Covert a sparse matrix into a dense np array
to_dense_array = lambda X: X.toarray() if sp.issparse(X) else np.asarray(X)

## Returns the data matrix or representation
extract_data_matrix = lambda adata,rep: adata.X if rep is None else adata.obsm[rep]

