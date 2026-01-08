# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 18:21:34 2025

@author: 18307
"""

import numpy as np

import feature_engineering
# from utils import utils_feature_loading
from utils import utils_visualization

# laplacian graph filtering
def apply_graph_laplacian_filtering(matrix, distance_matrix,
                                    filtering_params={'sigma': 0.1, 'k': 3,
                                                      'kernel_normalization': False,
                                                      'laplacian_normalization': True,
                                                      'alpha': 1, 'mode': 'highpass',
                                                      'lateral_mode': 'unilateral'}):
    matrix = np.array(matrix, dtype=float, copy=True)
    distance_matrix = np.array(distance_matrix, dtype=float, copy=True)

    sigma, k = filtering_params.get('sigma', 0.1), int(filtering_params.get('k', 3))
    if isinstance(sigma, (int, float)):
        pass  # use directly
    elif sigma == "mean_nonzero" or sigma is None:
        sigma = np.mean(distance_matrix[distance_matrix > 0])
    elif sigma == "knn_median":
        N = distance_matrix.shape[0]
        knn_means = []
        for i in range(N):
            dists = np.sort(distance_matrix[i][distance_matrix[i] > 0])
            if len(dists) >= k:
                knn_means.append(np.mean(dists[:k]))
        sigma = np.median(knn_means)
    else:
        raise ValueError(f"Unknown sigma mode: {sigma}")
    kernel_normalization = bool(filtering_params.get('kernel_normalization', False))
    laplacian_normalization = bool(filtering_params.get('laplacian_normalization', True))
    alpha, mode = float(filtering_params.get('alpha', 1)), str(filtering_params.get('mode', 'highpass')) # "lowpass" or "highpass"
    lateral_mode = str(filtering_params.get('lateral_mode', 'unilateral')) # 'unilateral' or 'bilateral'

    # Avoid division by zero
    distance_matrix = np.where(distance_matrix == 0, 1e-6, distance_matrix)

    # Step 2: Construct adjacency matrix W (Gaussian kernel)
    W = np.exp(-np.square(distance_matrix) / (2 * sigma ** 2))
    np.fill_diagonal(W, 0)

    if kernel_normalization:
        W = W / (W.sum(axis=1, keepdims=True) + 1e-12)
    
    # Step 3: Compute Laplacian matrix L
    D = np.diag(W.sum(axis=1))
    if laplacian_normalization:
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        L = np.eye(W.shape[0]) - d_inv_sqrt @ W @ d_inv_sqrt
    else:
        L = D - W

    # Step 4: Construct filter matrix
    I = np.eye(W.shape[0])
    if mode == 'lowpass':
        F = I - alpha * L
    elif mode == 'highpass':
        F = alpha * L
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Step 5: Apply filtering
    if lateral_mode == 'bilateral':
        filtered_matrix = F @ matrix @ F.T
    elif lateral_mode == 'unilateral':
        filtered_matrix = F @ matrix
    else:
        raise ValueError(f"Unknown lateral_mode: {lateral_mode}")

    return filtered_matrix

# spectral graph filtering
def apply_exp_graph_spectral_filtering(matrix, distance_matrix,
                                   filtering_params={'sigma': 0.1, 'k': 3,
                                                     'kernel_normalization': False,
                                                     'laplacian_normalization': True,
                                                     't': 1, 'mode': 'highpass',
                                                     'lateral_mode': 'unilateral'}):
    matrix = np.array(matrix, dtype=float, copy=True)
    distance_matrix = np.array(distance_matrix, dtype=float, copy=True)

    sigma, k = filtering_params.get('sigma', 0.1), int(filtering_params.get('k', 3))
    if isinstance(sigma, (int, float)):
        pass  # use directly
    elif sigma == "mean_nonzero" or sigma is None:
        sigma = np.mean(distance_matrix[distance_matrix > 0])
    elif sigma == "knn_median":
        N = distance_matrix.shape[0]
        knn_means = []
        for i in range(N):
            dists = np.sort(distance_matrix[i][distance_matrix[i] > 0])
            if len(dists) >= k:
                knn_means.append(np.mean(dists[:k]))
        sigma = np.median(knn_means)
    else:
        raise ValueError(f"Unknown sigma mode: {sigma}")
    kernel_normalization = filtering_params.get('kernel_normalization', False)
    laplacian_normalization = filtering_params.get('laplacian_normalization', True)
    t, mode = filtering_params.get('t', 1), filtering_params.get('mode', 'highpass').lower() # "lowpass" or "highpass"
    lateral_mode = str(filtering_params.get('lateral_mode', 'unilateral'))  # 'unilateral' or 'bilateral'

    # Step 1: Construct adjacency matrix W (Gaussian kernel)
    distance_matrix = np.where(distance_matrix == 0, 1e-6, distance_matrix)

    W = np.exp(-np.square(distance_matrix) / (2 * sigma ** 2))
    np.fill_diagonal(W, 0)

    if kernel_normalization:
        W = W / (W.sum(axis=1, keepdims=True) + 1e-12)

    # Step 2: Compute Laplacian matrix L
    D = np.diag(W.sum(axis=1))
    if laplacian_normalization:
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        L = np.eye(W.shape[0]) - d_inv_sqrt @ W @ d_inv_sqrt
    else:
        L = D - W

    # Step 3: Spectral decomposition of L
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Step 4: Construct exponential filter
    if mode == 'lowpass':
        h_lambda = np.exp(-t * eigenvalues)      # Low-pass: keep smooth components
    elif mode == 'highpass':
        h_lambda = 1 - np.exp(-t * eigenvalues)  # High-pass: remove smooth components
    else:
        raise ValueError("Invalid mode. Use 'lowpass' or 'highpass'.")

    H = eigenvectors @ np.diag(h_lambda) @ eigenvectors.T

    # Step 5: Apply filtering
    if lateral_mode == 'bilateral':
        filtered_matrix = H @ matrix @ H.T
    elif lateral_mode == 'unilateral':
        filtered_matrix = H @ matrix
    else:
        raise ValueError(f"Unknown lateral_mode: {lateral_mode}")

    return filtered_matrix

# generalized surface laplacian filtering
def apply_generalized_surface_laplacian_filtering(matrix, distance_matrix,
                                                  filtering_params={'sigma': 0.1, 'k': 3,
                                                                    'kernel_normalization': True,
                                                                    'knn': -1, 'residual_normalization': False,
                                                                    'symmetrize': True}):
    matrix = np.array(matrix, dtype=float, copy=True)
    distance_matrix = np.array(distance_matrix, dtype=float, copy=True)

    sigma, k = filtering_params.get('sigma', 0.1), filtering_params.get('k', 3)
    if isinstance(sigma, (int, float)):
        pass  # use directly
    elif sigma == "mean_nonzero" or sigma is None:
        sigma = np.mean(distance_matrix[distance_matrix > 0])
    elif sigma == "knn_median":
        N = distance_matrix.shape[0]
        knn_means = []
        for i in range(N):
            dists = np.sort(distance_matrix[i][distance_matrix[i] > 0])
            if len(dists) >= k:
                knn_means.append(np.mean(dists[:k]))
        sigma = np.median(knn_means)
    elif sigma == "median_heuristic":
        nonzero_dists = distance_matrix[distance_matrix > 0]
        sigma = np.median(nonzero_dists)
    else:
        raise ValueError(f"Unknown sigma mode: {sigma}")
    
    # print('sigma: ', sigma)
    
    kernel_normalization = bool(filtering_params.get('kernel_normalization', True))
    knn = int(filtering_params.get('knn', -1))
    residual_normalization = bool(filtering_params.get('residual_normalization', False))
    symmetrize = bool(filtering_params.get('symmetrize', True))

    distance_matrix = np.where(distance_matrix == 0, 1e-6, distance_matrix)

    # Step 1: Construct Gaussian kernel (W)
    W = np.exp(-np.square(distance_matrix) / (2 * sigma ** 2))
    np.fill_diagonal(W, 0)

    if kernel_normalization:
        W = W / (W.sum(axis=1, keepdims=True) + 1e-12)

    # 3) 可选 kNN 稀疏化（每行只保留 k 个最大权重）
    N = matrix.shape[0]
    if knn > 0 and knn < N-1:
        # 对每一行，保留最大的 knn 个非对角元素
        # 用 partition 实现 O(N) 选择，再零掉其余
        idx = np.argpartition(W, -knn, axis=1)[:, -(knn):]   # 每行 top-k 的列索引（无序）
        mask = np.zeros_like(W, dtype=bool)
        row_indices = np.arange(N)[:, None]
        mask[row_indices, idx] = True
        # 保证对称性：取 mask 或其转置的并集（避免破坏 W 的对称）
        mask = np.logical_or(mask, mask.T)
        W = np.where(mask, W, 0.0)

    # 4) 预计算行和
    r = W.sum(axis=1)

    # 5) 两次矩阵乘法（BLAS 加速）
    A = matrix @ W
    B = W @ matrix

    if residual_normalization:
        # 分子与分母
        num = A + B
        den = r[None, :] + r[:, None] - 2.0 * W  # shape (N,N)

        eps = 1e-12
        neighbor_avg = np.where(den > eps, num / den, matrix)
    else:
        neighbor_avg = A + B

    # 6) 滤波：M' = M - neighbor_avg
    matrix_filtered = matrix - neighbor_avg

    # 8) 可选对称化
    if symmetrize:
        matrix_filtered = 0.5 * (matrix_filtered + matrix_filtered.T)

    return matrix_filtered

# %% aplly filters on fcs
def fcs_filtering_common(fcs,
                         projection_params={"source": "auto", "type": "3d_euclidean"},
                         filtering_params={}, 
                         apply_filter='graph_laplacian_filtering',
                         visualize=False):
    
    # Step 1: Compute spatial distance matrix
    _, distance_matrix = feature_engineering.compute_distance_matrix('seed', projection_params=projection_params)
    distance_matrix = feature_engineering.normalize_matrix(distance_matrix)

    # Step 2: Apply filtering to each FC matrix
    if fcs.ndim == 2:
        if apply_filter == 'graph_laplacian_filtering' or 'glf':
            fcs_filtered = apply_graph_laplacian_filtering(fcs, distance_matrix, filtering_params)
        elif apply_filter == 'graph_spectral_filtering' or 'gsf':
            fcs_filtered = apply_exp_graph_spectral_filtering(fcs, distance_matrix, filtering_params)
        elif apply_filter == 'generalized_surface_laplacian_filtering' or 'gslf':
            fcs_filtered = apply_generalized_surface_laplacian_filtering(fcs, distance_matrix, filtering_params)
        
        if visualize:
            utils_visualization.draw_projection(fcs_filtered)
        
    elif fcs.ndim == 3:
        fcs_filtered = []
        
        if apply_filter == 'graph_laplacian_filtering' or 'glf':
            for fc in fcs:
                filtered = apply_graph_laplacian_filtering(fc, distance_matrix, filtering_params)
                fcs_filtered.append(filtered)
        
        elif apply_filter == 'graph_spectral_filtering' or 'gsf':
            for fc in fcs:
                filtered = apply_exp_graph_spectral_filtering(fc, distance_matrix, filtering_params)
                fcs_filtered.append(filtered)
        
        elif apply_filter == 'generalized_surface_laplacian_filtering' or 'gslf':
            for fc in fcs:
                filtered = apply_generalized_surface_laplacian_filtering(fc, distance_matrix, filtering_params)
                fcs_filtered.append(filtered)
        
        if visualize:
            average = np.mean(fcs_filtered, axis=0)
            utils_visualization.draw_projection(average)
        
    return np.stack(fcs_filtered)

# %% Usage
if __name__ == '__main__':
    # electrodes = utils_feature_loading.read_distribution('seed')['channel']
    
    # %% Distance Matrix
    _, distance_matrix_euc = feature_engineering.compute_distance_matrix(dataset="seed", 
                                                                        projection_params={"source": "auto", "type": "3d_euclidean"})
    distance_matrix_euc = feature_engineering.normalize_matrix(distance_matrix_euc)
    utils_visualization.draw_projection(distance_matrix_euc) # , xticklabels=electrodes, yticklabels=electrodes)
    
    _, distance_matrix_sph = feature_engineering.compute_distance_matrix(dataset="seed", 
                                                                        projection_params={"source": "auto", "type": "3d_spherical"})
    distance_matrix_sph = feature_engineering.normalize_matrix(distance_matrix_sph)
    utils_visualization.draw_projection(distance_matrix_sph) # , xticklabels=electrodes, yticklabels=electrodes)
    
    _, distance_matrix_gsp = feature_engineering.compute_distance_matrix(dataset="seed", 
                                                                         projection_params={"source": "auto", "type": "graph_shortest_path"})
    distance_matrix_gsp = feature_engineering.normalize_matrix(distance_matrix_gsp)
    utils_visualization.draw_projection(distance_matrix_gsp) # , xticklabels=electrodes, yticklabels=electrodes)
    
    _, distance_matrix_rd = feature_engineering.compute_distance_matrix(dataset="seed", 
                                                                        projection_params={"source": "auto", "type": "resistance_distance"})
    distance_matrix_rd = feature_engineering.normalize_matrix(distance_matrix_rd)
    utils_visualization.draw_projection(distance_matrix_rd) # , xticklabels=electrodes, yticklabels=electrodes)
    
    #
    _, distance_matrix_gsp1 = feature_engineering.compute_distance_matrix(dataset="seed", 
                                                                         projection_params={"source": "auto", "type": "graph_shortest_path",
                                                                                            "graph": {'k': 3}})
    distance_matrix_gsp1 = feature_engineering.normalize_matrix(distance_matrix_gsp1)
    utils_visualization.draw_projection(distance_matrix_gsp1) # , xticklabels=electrodes, yticklabels=electrodes)
    
    _, distance_matrix_rd1 = feature_engineering.compute_distance_matrix(dataset="seed", 
                                                                        projection_params={"source": "auto", "type": "resistance_distance",
                                                                                            "graph": {'k': 3}})
    distance_matrix_rd1 = feature_engineering.normalize_matrix(distance_matrix_rd1)
    utils_visualization.draw_projection(distance_matrix_rd1) # , xticklabels=electrodes, yticklabels=electrodes)
    
    # %% Connectivity Matrix
    # get sample and visualize sample
    # sample_averaged = utils_feature_loading.read_fcs_global_average('seed', 'pcc', 'gamma', sub_range=range(1, 16))
    # utils_visualization.draw_projection(sample_averaged)
    
    # # graph_laplacian_filtering; alpha = 1
    # cm_filtered = fcs_filtering_common(sample_averaged,
    #                                    projection_params={"source": "auto", "type": "3d_spherical"},
    #                                    filtering_params={'computation': 'graph_laplacian_filtering',
    #                                                      'alpha': 1, 'sigma': 'knn_median',
    #                                                      'lateral_mode': 'bilateral', 'normalized': False, 'reinforce': False}, 
    #                                    apply_filter='graph_laplacian_filtering',
    #                                    visualize=True)
    
    # #  exp_graph_spectral_filtering; t = 10
    # cm_filtered = fcs_filtering_common(sample_averaged,
    #                                    projection_params={"source": "auto", "type": "3d_spherical"},
    #                                    filtering_params={'computation': 'exp_graph_spectral_filtering', 
    #                                                      't': 10, 'mode': 'lowpass',
    #                                                      'normalized': False, 'reinforce': False}, 
    #                                    apply_filter='exp_graph_spectral_filtering',
    #                                    visualize=True)