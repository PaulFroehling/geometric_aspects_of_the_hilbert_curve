import numpy as np
from utils.evaluation_utils import (
    create_3d_scatter_plot,
    eval_knn_index_span,
    eval_k_index_neighbors,
    compute_hilbert_ranks_for_data,
    compute_z_ranks_for_data)


N_POINTS          = 100000
N_DIMS            = 3 
N_BITS            = 8
N_POINTS_PER_AXIS = 2**N_BITS
VOID_SIZE = 70
K = 10


def create_uniform_and_void_data(void_size:int) -> tuple[np.ndarray, np.ndarray]:
    """Create uniformly distributed data within the cube of a 3D Hilbert curve of size N_POINTS_PER_AXIS, N_DIMS
    It creates of cube with and witout a void. 

    Args:
        void_size (int): Size of the void being cut to the cube

    Returns:
        tuple[np.ndarray, np.ndarray]: Uniformly distributed points with and without a void
    """ 
    uniform_data = np.random.uniform(0,N_POINTS_PER_AXIS, (N_POINTS, N_DIMS))
    grid_center = np.array([N_POINTS_PER_AXIS/2, N_POINTS_PER_AXIS/2, N_POINTS_PER_AXIS/2])
    distances = np.linalg.norm(grid_center-uniform_data, axis=1)
   
    mask = (void_size <= distances)
    #mask = (distances > void_size) & (distances <= void_size + 30)
    uniform_with_void = uniform_data[mask]

    return uniform_data, uniform_with_void


def plot_heatmap(data:np.ndarray, all_spans:np.ndarray) -> None:
    """Plots the heatmap for set of data with measured distances

    Args:
        data (np.ndarray): Dataset
        all_spans (np.ndarray): Distance per Data point
    """    

    mask = data[:,0] > (N_POINTS_PER_AXIS / 2)

    create_3d_scatter_plot(data, np.array(all_spans))
    create_3d_scatter_plot(data[mask], all_spans[mask])


def growing_void_experiment() -> None:
    """Evaluates the pointwise knn-distance for samples of the dataset, considering hilbert indices
    """    
    unif_data, void_data = create_uniform_and_void_data(void_size=VOID_SIZE)
    h_ranks, _ = compute_hilbert_ranks_for_data(data=unif_data, n_dims=N_DIMS, n_bits=N_BITS)
    z_ranks, _ = compute_z_ranks_for_data(data=unif_data, n_dims=N_DIMS, n_bits=N_BITS)
    _, all_spans_h = eval_knn_index_span(data = unif_data, ranks = h_ranks, k = K)
    _, all_spans_z = eval_knn_index_span(data = unif_data, ranks = z_ranks, k = K)

    plot_heatmap(data = unif_data, all_spans = all_spans_h)
    plot_heatmap(data = unif_data, all_spans = all_spans_z)

    h_ranks_void, _ = compute_hilbert_ranks_for_data(data=void_data, n_dims=N_DIMS, n_bits=N_BITS)
    dists, sorted_data = eval_k_index_neighbors(void_data, h_ranks_void, k = 10)
    mask = sorted_data[:,0] > (N_POINTS_PER_AXIS / 2)
    create_3d_scatter_plot(sorted_data[mask], dists[mask])
    print(max(dists)-min(dists))

    # for i in range(50,N_POINTS_PER_AXIS):
    #     _, void_data = create_uniform_and_void_data(void_size = i)
    #     span_void, _, _ = eval_hilbert_knn_index_span(data=void_data, k=K, n_dims=N_DIMS, n_bits=N_BITS)
    #     print(f"Number of points: {len(void_data)} | Void size:{i} | Index span:{span_void/len(void_data)}")
    #     #print(f"Index span of the uniform distribution with void of size {i} not normalized:{span_void}")


growing_void_experiment()