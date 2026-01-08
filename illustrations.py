# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 19:09:12 2025

@author: 18307
"""
import numpy as np
import matplotlib.pyplot as plt

def draw_projection(sample_projection, title=None, xticklabels=None, yticklabels=None,
                    figsize=(2, 2)):
    """
    Visualizes data projections (common for both datasets).
    
    Parameters:
        sample_projection (np.ndarray): 2D or 3D matrix to visualize.
        title (str): Optional plot title.
        xticklabels (list): Optional list of x-axis labels.
        yticklabels (list): Optional list of y-axis labels.
        figsize (tuple): Optional size of the figure (width, height).
    """
    if title is None:
        title = "2D Matrix Visualization"
    
    def apply_axis_labels(ax, xticks, yticks):
        if xticks is not None:
            ax.set_xticks(range(len(xticks)))
            ax.set_xticklabels(xticks, rotation=90)
        if yticks is not None:
            ax.set_yticks(range(len(yticks)))
            ax.set_yticklabels(yticks)

    if sample_projection.ndim == 2:
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(sample_projection, cmap='viridis')
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        apply_axis_labels(ax, xticklabels, yticklabels)
        plt.tight_layout()
        plt.show()

    elif sample_projection.ndim == 3 and sample_projection.shape[0] <= 100:
        for i in range(sample_projection.shape[0]):
            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(sample_projection[i], cmap='viridis')
            plt.colorbar(im, ax=ax)
            ax.set_title(f"Channel {i + 1} Visualization")
            apply_axis_labels(ax, xticklabels, yticklabels)
            plt.tight_layout()
            plt.show()

    else:
        raise ValueError(f"The dimension of sample matrix for drawing is wrong, shape of sample: {sample_projection.shape}")

from utils import utils_feature_loading

de_gamma_sample = utils_feature_loading.read_cfs('seed', 'sub1ex1', 'de')['gamma']

electrodes = utils_feature_loading.read_distribution('seed')['channel']

from utils import utils_visualization

utils_visualization.draw_heatmap_1d(de_gamma_sample[0:99, 0:23].T, electrodes[0:23], figsize=(20,5))

# for i in range(0,10):
#     utils_visualization.draw_heatmap_1d(de_gamma_sample[i:i+1, 0:23].T, electrodes[0:23], [i], figsize=(0.2,5))

pcc_gamma_sample = utils_feature_loading.read_fcs('seed', 'sub1ex1', 'pcc')['gamma']

# utils_visualization.draw_projection(pcc_gamma_sample[0, 0:32, 0:32], None, electrodes[0:32], electrodes[0:32])
serie = np.arange(0, 8)
np.random.shuffle(serie)
serie=serie[0:8]
serie=np.sort(serie)
draw_projection(pcc_gamma_sample[0, serie][:, serie], None, electrodes[serie], electrodes[serie])

# import feature_engineering

# feature_engineering.compute_distance_matrix('seed', {'source': 'auto', 'type': '2d_flat', 'resolution': 19}, True)
# feature_engineering.compute_distance_matrix('seed', {'source': 'manual', 'type': '2d_flat', 'resolution': 19}, True, c='orange')

# feature_engineering.compute_distance_matrix('seed', {'source': 'auto', 'type': 'stereographic', 'resolution': 19}, True)