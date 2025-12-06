'''
Analysis of the indexing performance of the Hilbert curve using gaussian clusters of different isotropy (round vs multiple degrees of oval)
Isotropy is  analyzed under different turning-transformations of the SO(3) symmetry group, preserving axis orientations of a cubic grid. 
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.evaluation_utils import *


N_POINTS          = 1600
N_DIMS            = 3 
N_BITS            = 8
N_POINTS_PER_AXIS = 2**N_BITS
RESULTS_COLS      = ['Avg Isotropy','Std Isotropy', 'Avg Volume', 'Std Volume' , 'Avg Collisions', 'Span of Collisions', 'Avg Index Span', 'Span of Index Spans', 'index_spans']
RESULTS           = []


def create_covs_for_anisotropic_gaussian_clusters() -> np.ndarray:
    '''Create covarinace matrices for gaussian cluster of different isotropy / spread
    While x is fixed, the variance of y and z can vary. 
    Since the going over the full range of cov_y and cov_z will lead to similar cluster of different orientation
    Only covariance matrices will be created, until cov_y < cov_z
    
    Args:

    Returns:
        - diags (np.array): Diagonal matrices as covariance matrices
    '''
    cov_start = 0.1
    cov_end = 10
    num_steps = 20

    cov_x = 5
    cov_y = np.linspace(cov_start, cov_end, int(num_steps))
    cov_z = cov_y[::-1]
    diags = [np.diag([cov_x, cov_y[i], cov_z[i]]) for i in range(0,len(cov_y)) if cov_y[i] < cov_z[i]]

    return diags


def create_anisotropic_gaussian_clusters(gaussian_std_cluster:np.ndarray) -> np.ndarray:
    '''Creates more or less isotropic gaussian clusters by using cholesky decomposition.
    Using a gaussian cluster from a standard normal distribution as input, the decomposition reduces from LDL^T to LL^T since D is the unit matrix for std normal.
    The diagonal needs to be normed to create equal volume for all clusters. This can be achieved by using the third root (in 3D) of the determinant of the diagonal matrix.

    '''
    diags = create_covs_for_anisotropic_gaussian_clusters()
    anisotropic_gaussians = []
    for diag in diags:
        normed_diag = diag / (np.linalg.det(diag)**(1/N_DIMS))
        l = np.linalg.cholesky(normed_diag)
        anisotropic_gaussians.append(gaussian_std_cluster @ l.T)

    return np.array(anisotropic_gaussians)


def compute_global_max_spreads(datasets:np.ndarray) -> np.ndarray:
    '''Computes the maximum spreads for all dimensions for one or many datasets.
    Spread = max value of a dimension - min value of a dimension

    Args:
        - datasets (np.ndarray): List of datasets
    Returns:
        - global_max_spreads (np.ndarray): List of max ranges, one for each dimension
    '''

    max_vals = np.max(datasets, axis=1)
    min_vals = np.min(datasets, axis=1)
    ranges = max_vals - min_vals
    global_max_spreads = np.max(ranges, axis=0)
    
    return global_max_spreads


def scale_globally(data:np.ndarray, global_max_spreads:np.ndarray) -> np.ndarray:
    ''' Scale data to the SFC grid while preserving equal volumes of the clusters.
    This can be achieved by using the maximum spread of all clusters over all dimensions. 
    Intuition: A very anistropic cluster has a bigger spread over one axis than a more isotropic one, since volume is fixed.
    N_POINTS_PER_AXIS / maximum_spread is the ratio, by this spread needs to be scaled to be within the grid of the hilbert curve.
    All other values on different axes will also be scaled by this factor. Since it orientates on the biggest spread, the second biggest spread will also fall between 0 and N_POINTS_PER_AXIS
    By using this proceedure, scaling will preserve equal volumes among the cluster, since the scaling factors is used for all of them. 

    Args:
        -data (np.array): Data to be scaled
        -global_max_spreads: Maximum spreads over axes among one or multiple datasets
    Returns:
        -scaled_cluster (np.ndarray): Data scaled to N_POINTS_PER_AXIS
    '''
    maximum_spread = np.max(global_max_spreads) # Using the global maximum spread ensures that all clusters are scaled consistently
    glob_scaling_factor = (N_POINTS_PER_AXIS-1) / maximum_spread # relative to the cluster with the largest extent in any axis.
    min_val = np.min(data, axis = 0)
    scaled_data = (data - min_val) * glob_scaling_factor
 
    return np.array(scaled_data)


def plot_gaussian_cluster_variations(subplots_x:int, subplots_y:int, gaussians:np.ndarray):
    '''Used for plotting different shapes of created clusters.
    Args:
        -subplots_x (int): Number of plots on x-axis
        -subplots_y (int): Number of plots on y-axis
        -gaussians (np.ndarray): Gaussians to be plotted

    Returns:
        -
    '''
    _, axes = plt.subplots(subplots_x, subplots_y, figsize=(16, 20), subplot_kw={'projection':'3d'})
    axes = axes.flatten()
    for idx, ax in enumerate(axes):
        ax.scatter(gaussians[idx,:,0], gaussians[idx,:,1],gaussians[idx,:,2])
        ax.set_xlim(0, N_POINTS_PER_AXIS-1)
        ax.set_ylim(0, N_POINTS_PER_AXIS-1)
        ax.set_zlim(0, N_POINTS_PER_AXIS-1)
        ax.view_init(elev=0, azim=180)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    plt.tight_layout()
    plt.show()


def filter_rotations(rotations:np.ndarray) -> np.ndarray:
    ''' Not all rotations of the SO(3) provide meaningful input for this analysis.
    E.g. rotating the most anistropic cluster by 180 degree is basically identity.
    The concept is to filter by variance E.g. two rotations of a cluster with variances: (1, 3, 1) and (1,1,3) -> duplicate
    Since the variance of x won't be changed in the cluster creation process, filtering by x-values 
    and only keep rotations with an unseen x-variance, will lead to meaningful, unique transformations. 
    
    Args:
        -rotations (np.ndarray): Rotated versions of a cluster
    Returns:
        -filtered_rotations (np.ndarray): Rotations of a cluster without duplicates
    '''
    filtered_rotations = []
    used_x_variance = set()
    for rotation in rotations:
        variances = np.round(np.diagonal(np.cov(rotation, rowvar=False)), decimals=6)
        if variances[0] not in used_x_variance:
            filtered_rotations.append(rotation)
            used_x_variance.add(variances[0])

    return np.array(filtered_rotations)


def append_to_results(input_dict:dict) -> None:
    '''Appends a row of results (i.e. metrics calculated over all rotations of one cluster) to the results list.

    Args:
        - input_dict (np.ndarray): Dictionary of metrics

    Returns:
        -
    '''
    avg_iso = np.round(np.mean(input_dict['iso']), decimals=3)
    std_iso = np.round(np.std(input_dict['iso']), decimals=3)
    avg_idx_clusters = np.round(np.mean(input_dict['idx_clusters']), decimals=2)
    avg_vol = np.round(np.mean(input_dict['volumes']), decimals=2)
    std_vol = np.round(np.std(input_dict['volumes']), decimals=2)
    avg_collision = np.round(np.mean(input_dict['collision_rates']), decimals=3)
    collision_span = np.round(np.max(input_dict['collision_rates']) - np.min(input_dict['collision_rates']), decimals=2)
    avg_index_span = np.round(np.mean(input_dict['index_spans']), decimals=2)
    span_of_idx_spans = np.round((np.max(input_dict['index_spans']) - np.min(input_dict['index_spans'])), decimals=2)
    row = [avg_iso, std_iso, avg_vol, std_vol, avg_collision, collision_span, avg_index_span, span_of_idx_spans, avg_idx_clusters]
    RESULTS.append(row)


def create_results_df_and_save() -> None:
    '''Turns the RESULTS list to a pandas dataframe and saved it as excel file.

    Args:
        -
    Returns:
        -
    '''
    results_df = pd.DataFrame(RESULTS, columns=RESULTS_COLS)
    results_df.to_excel('data_out/gaussians/results.xlsx','main results')
    print(results_df)
    


def gaussian_isotropy_experiment(gaussian_cluster:np.ndarray):
    '''Analyzes to which degree the performance of a space filling curve (SFC) like the Hilbert curve is sensitive to changes in isotropy and orientation.
    Isotropy means: A gaussian cluster is more spread on one axis, than on another.
    Orientation means: Turning the cluster while preserving symmetry of the grid.
    
    - The experiment starts with a gaussian cluster and stretches it, while preserving the volume of the cluster (otherwise the density of points would change
        for the same amount of points and so the closeness of indices for neighboring points. Different clusters with different spread/isotropy are computed
    - To also analyze rotation sensitivity (especially for anistropic clusters), transformations from the SO(3) group are used to rotate each cluster
        Not all transformations are kept because they lead to to symmetric results also for the anisotropic ones. E.g. turning a 'cigar-shaped' cluster by 180 doesn't give new insights
    - For analyzing the performance of the SFC, for each point in each rotated cluster, the index spread for the k next neighbors is analyzed.

    Args:
        -gaussian_cluster (np.ndarray): Gaussian cluster with standard normal distribution

    Returns:
        -
    '''
    #1) Creates more or less anisotropic clusters based on a gaussian std norm cluster and scale them to a fixed grid size
    aniso_gaussians = create_anisotropic_gaussian_clusters(gaussian_std_cluster=gaussian_cluster)
    global_max_spreads = compute_global_max_spreads(datasets=aniso_gaussians)
    aniso_gaussians = np.array([scale_globally(data=cluster, global_max_spreads=global_max_spreads) for cluster in aniso_gaussians])
    aniso_gaussians = center_data(data=aniso_gaussians, n_points_per_axis=N_POINTS_PER_AXIS)

    #2) Compute rotation matrices of the SO(3)/Octaeder group
    so3_rotation_matrices = create_so3_transformation_matrices()

    print(f'Experiment for {len(aniso_gaussians)} clusters with {N_POINTS} points each, rotated by 3 transformations of SO(3) group, being unique for our clusters')

    #3) Create a knn setup: Analyze the spread of (hilbert) indices of the k neighbors for each point for each cluster and each rotation
    k = 6
    for i, cluster in enumerate(aniso_gaussians):
        #4)For each cluster: Rotate clusters, center the result and filter duplicate/symmetric results after rotation
        rotations = rotate_dataset_with_matrices(data=cluster, rotation_matrices=so3_rotation_matrices)
        rotations = center_data(data=rotations, n_points_per_axis=N_POINTS_PER_AXIS)
        rotations = filter_rotations(rotations=rotations)

        if i==0:
            plot_gaussian_cluster_variations(subplots_x=2, subplots_y=5, gaussians=aniso_gaussians)

        #5) For each rotation: Calculate volume (for consistency check), isotropy (also for consistency check), average index span over all 'neighborhood' and collision rates
        metrics = {'iso':[], 'volumes':[], 'avg_spans':[], 'collision_rates':[], 'index_spans':[], 'idx_clusters': []}
        for rotation in rotations:
            h_ranks, h_idc = compute_hilbert_ranks_for_data(data=rotation, n_dims=N_DIMS, n_bits=N_BITS)
            avg_span,_ = eval_knn_index_span(data=rotation,ranks=h_ranks, k=k)
            kkn_idx_clusters = eval_knn_clusters(data=rotation, ranks=h_ranks, k=k)
            collision_rate = compute_collision_rate(indices=h_idc)
            metrics['index_spans'].append(avg_span)
            metrics['idx_clusters'].append(np.mean(kkn_idx_clusters))
            metrics['collision_rates'].append(collision_rate)
            metrics['iso'].append(compute_isotropy_measure(data=rotation))
            metrics['volumes'].append(measure_volume_of_cluster(data=rotation))
        
        append_to_results(metrics)  
    create_results_df_and_save()

gaussian_isotropy_experiment(np.random.normal(0,1,(N_POINTS,3)))
