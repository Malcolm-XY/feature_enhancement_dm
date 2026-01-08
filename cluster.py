# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 01:37:49 2026

@author: 18307
"""

import numpy as np
import pandas as pd

# %% Cluster
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def hierarchical_clustering(distance_matrix, threshold=None, channel_index=None, parse=False, verbose=False):
    """
    Perform hierarchical clustering to group signals based on a correlation matrix.

    :param correlation_matrix: ndarray
        Correlation coefficient matrix (n x n).
    :param threshold: float or None
        Dissimilarity threshold for clustering. If None, an automatic threshold is calculated 
        based on the average dissimilarity (default: None).
    :param parse: bool
        If True, parse the clusters into lists of grouped indices.
    :param verbose: bool
        If True, print additional information such as the number of clusters (default: False).
    :return: 
        clusters: ndarray
            Cluster labels for each signal.
        parsed_clusters: list (optional)
            Parsed clusters as groups of indices (if `parse=True`).
    """
    # Compute the distance matrix
    np.fill_diagonal(distance_matrix, 0)  # Set diagonal to 0 (self-distance)

    # Convert to condensed distance matrix for linkage
    condensed_dist = squareform(distance_matrix, checks=False)

    # Automatically determine the threshold if not provided
    if threshold is None:
        threshold = np.mean(condensed_dist)  # Use the mean of the condensed distance matrix
        if verbose:
            print(f"Automatically determined threshold: {threshold:.4f}")

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method='average')
    clusters = fcluster(linkage_matrix, threshold, criterion='distance')
    
    # Parse clusters into groups of indices if required
    parsed_clusters = None
    if parse:
        parsed_clusters = {}
    
        for idx, cluster_id in enumerate(clusters):
            if cluster_id not in parsed_clusters:
                parsed_clusters[cluster_id] = []
    
            value = channel_index[idx] if channel_index is not None else idx
            parsed_clusters[cluster_id].append(value)
    
        parsed_clusters = list(parsed_clusters.values())
        
        if channel_index is not None:
            clusters = pd.DataFrame({'channel': channel_index, 'cluster_idx': clusters})
        
        if verbose:
            print(f"The number of clusters: {len(parsed_clusters)}")
    
    return (clusters, parsed_clusters) if parse else clusters

def plot_3d_channels(distribution, cluster_idx):
    """
    绘制 3D 通道分布，并按组用颜色区分。
    :param distribution: 包含 x, y, z 坐标和分组信息的 DataFrame
    """
    distribution['group'] = cluster_idx
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 获取唯一组标签
    unique_groups = distribution['group'].unique()

    colormap = plt.colormaps['viridis']  # 获取调色盘
    colors = [colormap(i / len(unique_groups)) for i in range(len(unique_groups))]  # 动态分配颜色
    
    for idx, group in enumerate(unique_groups):
        group_data = distribution[distribution['group'] == group]
        ax.scatter(
            group_data['x'], group_data['y'], group_data['z'],
            label=f"Group {group}",
            color=colors[idx],  # 通过索引访问颜色
            s=50  # 点的大小
            )
        # 添加文本标签
        for _, row in group_data.iterrows():
            ax.text(row['x'], row['y'], row['z'], row['channel'], fontsize=8)

    ax.set_title("3D Channel Distribution with Groups")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

from scipy.spatial import ConvexHull
def plot_3d_channels_(distribution, clusters):
    """
    绘制 3D 通道分布，并按组用颜色区分，同时圈出相同组的点。
    :param distribution: 包含 x, y, z 坐标和分组信息的 DataFrame
    """
    distribution['group'] = clusters
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 获取唯一组标签
    unique_groups = distribution['group'].unique()
    colormap = plt.colormaps['tab10']  # 获取调色盘
    colors = [colormap(i / len(unique_groups)) for i in range(len(unique_groups))]  # 动态分配颜色

    for idx, group in enumerate(unique_groups):
        group_data = distribution[distribution['group'] == group]
        # 绘制组内的点
        ax.scatter(
            group_data['x'], group_data['y'], group_data['z'],
            label=f"Group {group}",
            color=colors[idx],
            s=50  # 点的大小
        )
        # 添加文本标签
        for _, row in group_data.iterrows():
            ax.text(row['x'], row['y'], row['z'], row['channel'], fontsize=8)

        # 绘制组的边界（凸包）
        if len(group_data) >= 4:  # 凸包要求至少 4 个点
            points = group_data[['x', 'y', 'z']].to_numpy()
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                ax.plot_trisurf(
                    points[:, 0], points[:, 1], points[:, 2],
                    triangles=[simplex],
                    color=colors[idx],
                    alpha=0.2,  # 透明度
                    edgecolor='gray'
                )

    ax.set_title("3D Channel Distribution with Groups")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

# %% Feature Fussion
def construct_inter_cluster_mask(cluster_idx):
    length = len(cluster_idx)
    
    inter_cluster_mask = np.zeros([length, length])
    for i in range(length):
        for j in range(length):
            if cluster_idx[i] != cluster_idx[j]:
                inter_cluster_mask[i][j] = 1
            elif cluster_idx[i] == cluster_idx[j]:
                inter_cluster_mask[i][j] = 0
    
    return inter_cluster_mask

import feature_engineering
def reconstruct_fn(fn_basis, fn_modifier, inter_cluster_mask, params={'alpha': 0, 'beta': 0, 'scale': (0, 1)}):
    alpha, beta = params.get('alpha', 0.5), params.get('beta', 0.5)
    scale = params.get('scale', (0, 1))
    
    fn_basis = feature_engineering.normalize_matrix(fn_basis, 'minmax', param={'target_range': scale})
    fn_modifier = feature_engineering.normalize_matrix(fn_modifier, 'minmax', param={'target_range': scale})
    
    inter_cluster_mask = feature_engineering.normalize_matrix(inter_cluster_mask, 'minmax', param={'target_range': (0, 1)})
    inter_cluster_mask_i = 1 - inter_cluster_mask
    
    # global operater_basis
    # global operater_modifier
    
    mask_basis = inter_cluster_mask + alpha * inter_cluster_mask_i
    mask_modifier = inter_cluster_mask_i + beta * inter_cluster_mask
    
    fn_basis_ = fn_basis * mask_basis
    fn_modifier_ = fn_modifier * mask_modifier
    
    fn_ = fn_basis_ + fn_modifier_
    
    return fn_, fn_basis_, fn_modifier_
    
# %% Usage
if __name__ == '__main__':
    from utils import utils_feature_loading
    from utils import utils_visualization
    # cluster
    electrodes, dm = feature_engineering.compute_distance_matrix('seed', projection_params={'source': 'auto', 'type': '3d_euclidean'})
    clusters, parsed_clusters = hierarchical_clustering(dm, 70, electrodes, parse=True, verbose=True)
    
    distribution = utils_feature_loading.read_distribution('seed')
    
    plot_3d_channels_(distribution, clusters['cluster_idx'])
    plot_3d_channels(distribution, clusters['cluster_idx'])
    
    # feature fussion
    inter_cluster_mask = construct_inter_cluster_mask(clusters['cluster_idx'])
    utils_visualization.draw_projection(inter_cluster_mask)
    
    example_pcc = utils_feature_loading.read_fcs('seed', 'sub1ex1', 'pcc')['alpha']
    example_plv = utils_feature_loading.read_fcs('seed', 'sub1ex1', 'plv')['alpha']
    
    fn, fn_basis, fn_modifier = reconstruct_fn(example_pcc, example_plv, inter_cluster_mask,
                                               params={'alpha': 0, 'beta': 0, 'scale': (0, 1)})
    
    utils_visualization.draw_projection(example_pcc[0], 'pcc')
    utils_visualization.draw_projection(fn_basis[0], "pcc'")
    utils_visualization.draw_projection(example_plv[0], 'plv')
    utils_visualization.draw_projection(fn_modifier[0], "plv'")
    utils_visualization.draw_projection(fn[0], 'pcc & plv')
    
    avg_pcc = utils_feature_loading.read_fcs_global_average('seed', 'pcc')['alpha']
    avg_plv = utils_feature_loading.read_fcs_global_average('seed', 'plv')['alpha']
    
    fn_, fn_basis_, fn_modifier_ = reconstruct_fn(avg_pcc, avg_plv, inter_cluster_mask,
                                                  params={'alpha': 0, 'beta': 0, 'scale': (0, 1)})
    
    utils_visualization.draw_projection(avg_pcc, 'pcc')
    utils_visualization.draw_projection(fn_basis_, "pcc'")
    utils_visualization.draw_projection(avg_plv, 'plv')
    utils_visualization.draw_projection(fn_modifier_, "plv'")
    utils_visualization.draw_projection(fn_, 'pcc & plv')
