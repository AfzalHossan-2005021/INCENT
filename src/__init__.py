from .core import (
    pairwise_align,
    hierarchical_pairwise_align,
    align_multiple_slices
)

from .evaluation import (
    label_transfer_accuracy,
    foscttm,
    expression_transfer_corr,
    calculate_neighborhood_dissimilarity_cost,
    calculate_gene_expression_dissimilarity,
    cell_type_matching,
    benchmark_method,
    evaluate_alignment,
    calculate_performance_metrics,
)

from .visualize import (
    stack_slices_pairwise,
    visualize_alignment,
    visualize_3d_stack,
)

from .synthesize import (
    InteractiveCropSelector,
    create_interactive_rectangular_portion,
    select_rectangular_portion_blocking,
    preview_crop,
)

from .perturb import (
    simulate_adjacent_slice,
)

from .tuning import (
    select_alignment_weights,
    make_self_alignment_instances,
    simplex_grid,
    gpu_available,
)

__all__ = [
    # core alignment
    'pairwise_align',
    'hierarchical_pairwise_align',
    'align_multiple_slices',
    # primary metrics
    'label_transfer_accuracy',
    'foscttm',
    'expression_transfer_corr',
    'evaluate_alignment',
    'calculate_performance_metrics',
    # supplementary metrics
    'calculate_neighborhood_dissimilarity_cost',
    'calculate_gene_expression_dissimilarity',
    'cell_type_matching',
    # utilities
    'benchmark_method',
    # visualization
    'stack_slices_pairwise',
    'visualize_alignment',
    'visualize_3d_stack',
    # data preparation
    'InteractiveCropSelector',
    'create_interactive_rectangular_portion',
    'select_rectangular_portion_blocking',
    'preview_crop',
    'simulate_adjacent_slice',
    # weight selection
    'select_alignment_weights',
    'make_self_alignment_instances',
    'simplex_grid',
    'gpu_available',
]
