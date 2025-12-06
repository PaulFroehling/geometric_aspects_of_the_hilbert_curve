import numpy as np
from sklearn.neighbors import NearestNeighbors
from hilbert import encode
from zCurve import interlace
import matplotlib.pyplot as plt
from matplotlib import cm 
from matplotlib.colors import Normalize
from scipy.spatial.distance import pdist




def create_3d_scatter_plot(data:np.ndarray, index_ranges:None) -> None:
    #plt.style.use('dark_background')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if index_ranges is not None:
        
        cmap = cm.get_cmap('hot')
        norm = Normalize(vmin=min(index_ranges), vmax=max(index_ranges))
        colors = cmap(norm(index_ranges))
    else:
        colors = 'blue'
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors, marker='o', alpha =1.0, s=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()



def measure_volume_of_cluster(data:np.ndarray) -> float:
    ''' Measures the volume of a cluster (box around a cluster) by calculating the value ranges in x,y and z and multiplying them.

    Args:
        - data (np.ndarray): Data of which the volume needs to be computes
    Returns:
        - volume (float): Volume of the data (volume of the box around the data)
    '''
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    local_ranges = max_val - min_val
    volume = np.prod(local_ranges)

    return volume


def center_data(data:np.ndarray, n_points_per_axis:int) -> np.ndarray:
    '''Centers a given dataset to the middle of a grid with n_points_per_axis

    Args:
        - data (np.ndarray): Data to be centered
        - n_points_per_axis (int): Points per axis of a grid
    Returns
        - centered_data (np.ndarray): Centered data
    '''
    centered_data = [dataset + [(n_points_per_axis-1)/2]*3 - np.mean(dataset, axis=0) for dataset in data]
    return np.array(centered_data)



def compute_isotropy_measure(data:np.ndarray) -> float:
    '''Isotropy is defined (in this measure) as the ratio between the greatest and smallest eigenvalue 
    of the covariance matrix of a dataset. 1 minus this ratio will scale it from 0 to 1 (0 completely anistropic), (1 isotropic)

    Args:
        - data (np.ndarray): Dataset for which the isotropy needs to be calculated

    Returns
        -isotropy (float): Isotropy of the dataset. Ranges from 0 to 1
    '''
    cov = np.cov(data, rowvar=False)
    eig_vals_sorted = np.sort(np.linalg.eigvals(cov))

    isotropy = 1/(eig_vals_sorted[-1] / eig_vals_sorted[0])

    return isotropy


def get_rotation_matrix_for_unit_vector(uv:np.ndarray, alpha:int) -> np.ndarray:
    ''' Defines a rotation matrix for a given unit vector (uv) and an angle alpha i.e. the rotation angle.

    Args:
        - uv (np.ndarray): Unit vector for rotation transformation
        - alpha (float)  : Rotation angle
    
    Returns:
        - rotation_matrix (np.ndarray): Final rotation matrix
    '''
    uv /= np.linalg.norm(uv)
    alpha_rad = np.deg2rad(alpha)
    cos = np.cos(alpha_rad)
    sin = np.sin(alpha_rad)
    uv1, uv2, uv3 = uv
    rotation_matrix = [[uv1**2 * (1 - cos) + cos, uv1 * uv2 * (1 - cos) - uv3 * sin, uv1 * uv3 * (1-cos) + uv2 * sin],
                       [uv2 * uv1 * (1 - cos) + uv3 * sin, uv2**2 * (1 - cos) + cos, uv2 * uv3 * (1 - cos) - uv1 * sin],
                       [uv3 * uv1 * (1 - cos) - uv2 * sin, uv3 * uv2 * (1 - cos) + uv1 * sin, uv3**2 * (1-cos) + cos]]  

    return np.array(rotation_matrix)


def get_so3_unit_vectors() -> dict:
    ''' Provides the unit vectors for the rotations of the SO(3) symmetry group
    
    Args:
        -
    Returns:
        - uvs (dict): Unit vectors
    '''
    fourfold  = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    threefold = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]]
    twofold   = [[1,1,0], [1,-1,0], [1,0,1], [1,0,-1], [0,1,1], [0,1,-1]]
    uvs = {"fourfold":fourfold, "threefold": threefold, "twofold": twofold}

    return uvs


def create_so3_transformation_matrices() -> np.ndarray:
    '''Creates all transformation matrices for the rotations of SO(3) symmetry group.

    Args:
        -
    Returns:
        - rotation_matrices(np.ndarray): Array of rotation matrices for all SO(3) rotations
    '''
    uvs = get_so3_unit_vectors()
    rotation_matrices = []
    for uv in uvs["fourfold"]:
        for alpha in [90, 180, 270]:
            rotation_matrices.append(get_rotation_matrix_for_unit_vector(uv, alpha))

    for uv in uvs["threefold"]:  
        for alpha in [120, 240]:
            rotation_matrices.append(get_rotation_matrix_for_unit_vector(uv, alpha))

    for uv in uvs["twofold"]:  
             rotation_matrices.append(get_rotation_matrix_for_unit_vector(uv, 180))

    return np.array(rotation_matrices)


def rotate_dataset_with_matrices(data:np.ndarray, rotation_matrices:np.ndarray) -> np.ndarray:
    '''Rotates a dataset with a given set of rotation matrix.
    Args:
        - data (np.ndarray): Data to be rotated
        - rotation_matrices (np.ndarray): Array of rotation matrices
    Returns:
        - data_rotations (np.ndarray): Variants of the input data, rotated by the given matrices    
    '''
    data_rotations = [data @ rm for rm in rotation_matrices]
    data_rotations.append(data)

    return np.array(data_rotations)


def compute_collision_rate(indices:np.ndarray) -> float:
    '''The collision rate shows the percentage of points, that have a non-unique index. 

    Args:
        - hilbert_indices (np.ndarray): Array of indices
    Returns:
        - collision_rate (float): 1 - n_unique_values/n_indices
    '''
    collision_rate = 1 - len(set(indices)) / len(indices) 
    return collision_rate

def compute_ranks_from_indices(incs:np.ndarray) -> np.ndarray:
    ''' Argsort needs to be set to stable. If two points have the same hilbert index, the order of the initial list is preserved.
    '''
    ranks = np.argsort(np.argsort(incs, kind='stable')) # Stable retains list order for equal entries of equal value

    return ranks

def compute_z_ranks_for_data(data:np.ndarray, n_dims:int, n_bits:int):
    data = [list(map(int, datapoint)) for datapoint in data]
    z_idcs = np.array([interlace(*datapoint, dims=int(n_dims), bits_per_dim=int(n_bits)) for datapoint in data])
    z_ranks = compute_ranks_from_indices(z_idcs)
    
    return z_ranks, z_idcs


def compute_hilbert_ranks_for_data(data:np.ndarray, n_dims:int, n_bits:int) -> np.ndarray:
    h_idcs = encode(np.floor(data).astype(np.int32), num_dims=n_dims, num_bits=n_bits)
    h_ranks = compute_ranks_from_indices(h_idcs)
    
    return h_ranks, h_idcs


def eval_knn_index_span(data:np.ndarray, ranks:np.ndarray,  k:int) -> list[float, list]:
    '''Evaluates the span of hilbert indices within the set of k nearest neighbors of each point. 
    n_neighbors = k+1 because, the list of knns contains the point itself. 

    Args:
        - data   (np.ndarray): Data to evaluate the hilbert curve on
        - k      (int)       : Number of neighbors among the hilbert indices are compared

    Returns:
        [average_spans, collision_rate] (list): The average span (max-min) with neigborhoods of k around each point, collision_rate: Collision rate for the given dataset
    '''
    knns = NearestNeighbors(n_neighbors = k+1).fit(data)
    _, nbrs_list = knns.kneighbors(data)
    average_spans = []
    for nbrs in nbrs_list:
        nbrs_ranks = ranks[nbrs[1:]]
        average_spans.append(np.max(nbrs_ranks) - np.min(nbrs_ranks))

    return np.mean(average_spans), np.array(average_spans)


def eval_k_index_neighbors(data:np.ndarray, ranks:np.ndarray, k:int) -> list[float, list]:
    sorted_pairs = sorted(list(zip(ranks, data)), key=lambda x:x[0])
    _, data_sorted = list(zip(*sorted_pairs))
    data_sorted = np.array(data_sorted)
    n = len(data)

    windows = [data_sorted[max(0, i-k):min(n, i+k)] for i in range(n)]
    dists = [pdist(window) for window in windows]
    spans = np.array([max(dist_window) - min(dist_window) for dist_window in dists])
    avg_spans = np.mean(spans)
    return spans, np.array(data_sorted)


def eval_knn_clusters(data:np.ndarray, ranks:np.ndarray,  k:int) -> list[float, list]:
    '''Evaluates the span of hilbert indices within the set of k nearest neighbors of each point. 
    n_neighbors = k+1 because, the list of knns contains the point itself. 

    Args:
        - data   (np.ndarray): Data to evaluate the hilbert curve on
        - k      (int)       : Number of neighbors among the hilbert indices are compared

    Returns:
        [average_spans, collision_rate] (list): The average span (max-min) with neigborhoods of k around each point, collision_rate: Collision rate for the given dataset
    '''
    knns = NearestNeighbors(n_neighbors = k+1).fit(data)
    _, nbrs_list = knns.kneighbors(data)
    clusters = []
    for nbrs in nbrs_list:
        nbrs_ranks = np.sort(ranks[nbrs[1:]])
        element_wise_diffs = nbrs_ranks[1:] - nbrs_ranks[:-1]
        n_clusters = len(np.where(element_wise_diffs > 1)[0])
        clusters.append(n_clusters)

    return np.array(clusters)


