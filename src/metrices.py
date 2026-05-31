import numpy as np
import torch

from .core import (
    calculate_cell_type_mismatch,
    calculate_gene_expression_cosine_distance,
    calculate_neighborhood_dissimilarity,
)
from .utils import select_backend


def calculate_neighborhood_similarity(js_dist_neighborhood, pi):
    """
    Calculate neighborhood similarity cost for a given alignment mapping.
    
    Uses element-wise multiplication: sum all weighted distances across the mapping.
    Equivalent to INCENT.py's initial objective calculation for all dissimilarity types.
    
    Args:
        js_dist_neighborhood: Jensen-Shannon distance matrix of neighborhood distributions.
        pi: Alignment mapping matrix (either uniform G or optimal transport solution).
    
    Returns:
        neighborhood_similarity: Weighted neighborhood dissimilarity cost.
    """
    return np.sum(js_dist_neighborhood * pi)


def cell_type_matching(cell_type_mismatch, pi_mat):
    """
    Compute cell-type matching percentage from a pre-computed mismatch matrix.
    
    Args:
        cell_type_mismatch: Binary mismatch matrix (1 = mismatch, 0 = match) from calculate_cell_type_mismatch().
        pi_mat: Alignment mapping matrix (probabilistic transport plan).
    
    Returns:
        Percentage of transported mass representing cell-type matches (0-100).
    """
    M_match = 1 - cell_type_mismatch
    expected_matches = np.sum(M_match * pi_mat)
    total_mass = np.sum(pi_mat)

    if total_mass > 0:
        return (expected_matches / total_mass)
    return 0.0


def calculate_gene_expression_similarity(cosine_dist_gene_expr, pi):
    """
    Calculate gene expression similarity cost for a given alignment mapping.
    
    Uses element-wise multiplication: sum of weighted gene expression distances.
    Matches INCENT.py's calculation for both initial and final objectives.
    
    Args:
        cosine_dist_gene_expr: Cosine distance matrix of gene expression profiles.
        pi: Alignment mapping matrix (either uniform G or optimal transport solution).
    
    Returns:
        gene_expression_similarity: Weighted gene expression dissimilarity cost.
    """
    return np.sum(cosine_dist_gene_expr * pi)


def calculate_performance_metrics(final_pi, init_pi=None, js_dist_neighborhood=None, cosine_dist_gene_expr=None,
                               cell_type_mismatch=None, sliceA=None, sliceB=None, use_rep=None, radius=100.0, use_gpu=True):
    """
    Calculate all similarity metrics for alignment quality assessment.
    
    **Note:** Neighborhood similarity uses element-wise multiplication (sum over all mapping entries).
    This matches INCENT.py's initial objective calculation for all dissimilarity types (JSD/MSD/cosine).
    For specialized metrics like INCENT.py's final JSD objective (argmax per row), compute separately.
    
    Args:
        final_pi: Final optimal transport alignment mapping (required).
        init_pi: Initial alignment mapping (optional). If None, uses uniform distribution.
        js_dist_neighborhood: Jensen-Shannon distance matrix of neighborhood distributions (optional).
                             If not provided and sliceA, sliceB, radius are given, will be calculated.
        cosine_dist_gene_expr: Cosine distance matrix of gene expression profiles (optional).
                              If not provided and sliceA, sliceB, use_rep are given, will be calculated.
        sliceA: First slice for calculating missing distance matrices (optional).
        sliceB: Second slice for calculating missing distance matrices (optional).
        use_rep: Representation key for gene expression (optional, used with cosine_dist_gene_expr calculation).
        radius: Radius for neighborhood calculation (optional, used with js_dist_neighborhood calculation).
    
    Returns:
        Dictionary with keys: 'initial_obj_neighbor', 'initial_obj_gene', 
                              'final_obj_neighbor', 'final_obj_gene',
                              'initial_cell_type_match', 'final_cell_type_match'
                              
    Raises:
        ValueError: If required parameters for distance calculation are missing.
    """
    # Use uniform distribution if init_pi not provided
    if init_pi is None:
        init_pi = np.ones(final_pi.shape) / (final_pi.shape[0] * final_pi.shape[1])

    # Calculate js_dist_neighborhood if not provided
    if js_dist_neighborhood is None:
        if sliceA is None or sliceB is None:
            raise ValueError("sliceA and sliceB must be provided to calculate js_dist_neighborhood")

        # Calculate neighborhood dissimilarity using the provided slices and radius
        use_gpu, nx = select_backend(use_gpu=use_gpu, gpu_verbose=False)
        js_dist_neighborhood = calculate_neighborhood_dissimilarity(sliceA, sliceB, radius, nx=nx, data_type=np.float32, eps=1e-6)

        # Convert to numpy if necessary
        if isinstance(js_dist_neighborhood, torch.Tensor):
            js_dist_neighborhood = js_dist_neighborhood.detach().cpu().numpy()

    # Calculate cosine_dist_gene_expr if not provided
    if cosine_dist_gene_expr is None:
        if sliceA is None or sliceB is None:
            raise ValueError("sliceA and sliceB must be provided to calculate cosine_dist_gene_expr")
        cosine_dist_gene_expr = calculate_gene_expression_cosine_distance(sliceA, sliceB, use_rep)

    # Calculate cell-type matching metrics if slices are provided
    if cell_type_mismatch is None:
        if sliceA is None or sliceB is None:
            raise ValueError("sliceA and sliceB must be provided to calculate cell_type_matching_percentage")
        cell_type_mismatch = calculate_cell_type_mismatch(sliceA, sliceB)

    results = {}

    # Calculate neighborhood similarities
    results['initial_obj_neighbor'] = calculate_neighborhood_similarity(js_dist_neighborhood, init_pi)
    results['final_obj_neighbor'] = calculate_neighborhood_similarity(js_dist_neighborhood, final_pi)

    # Calculate gene expression similarities
    results['initial_obj_gene'] = calculate_gene_expression_similarity(cosine_dist_gene_expr, init_pi)
    results['final_obj_gene'] = calculate_gene_expression_similarity(cosine_dist_gene_expr, final_pi)

    # Calculate cell-type matching percentages
    results['initial_cell_type_match'] = cell_type_matching(cell_type_mismatch, init_pi)
    results['final_cell_type_match'] = cell_type_matching(cell_type_mismatch, final_pi)

    # Display results in a formatted table
    Title = "ALIGNMENT QUALITY METRICS"
    print("\n" + "="*80)
    print(" " * ((80 - len(Title)) // 2) + Title)
    print("="*80)
    print(f"{' Metric':<40} {'Initial':<12} {'Final':<12} {'Improvement':<12}")
    print("-"*80)
    print(f"{' Neighborhood Dissimilarity (JSD)':<40} {results['initial_obj_neighbor']:<12.6f} {results['final_obj_neighbor']:<12.6f} {(results['initial_obj_neighbor'] - results['final_obj_neighbor']) / results['initial_obj_neighbor'] * 100:>10.2f}%")
    print(f"{' Gene Expression Dissimilarity (Cosine)':<40} {results['initial_obj_gene']:<12.6f} {results['final_obj_gene']:<12.6f} {(results['initial_obj_gene'] - results['final_obj_gene']) / results['initial_obj_gene'] * 100:>10.2f}%")
    print(f"{' Cell-type Correspondence (%)':<40} {results['initial_cell_type_match']*100:<12.2f} {results['final_cell_type_match']*100:<12.2f} {(results['final_cell_type_match'] - results['initial_cell_type_match']) / results['initial_cell_type_match'] * 100:>10.2f}%")
    print("="*80 + "\n")

    return results
