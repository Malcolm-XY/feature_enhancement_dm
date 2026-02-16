# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 17:49:12 2025

@author: usouu
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% Matrix Plot
# %% Heatmap matrix plot
def matrix_plot(accuracy_data, lambda_values, sigma_values, fontsize=12):
    vmin, vmax = 60, 95
    fig, axes = plt.subplots(1, 4, figsize=(15, 5), constrained_layout=True)
    axes = axes.flatten()
    sr_keys = list(accuracy_data.keys())

    mappable = None

    for i, sr in enumerate(sr_keys):
        ax = axes[i]
        acc_matrix = accuracy_data[sr]
        show_xlabel = True # (i >= 4)
        show_ylabel = (i % 4 == 0)

        heatmap = sns.heatmap(
            acc_matrix,
            ax=ax,
            xticklabels=sigma_values if show_xlabel else False,
            yticklabels=lambda_values if show_ylabel else False,
            cmap='coolwarm',
            annot=True,
            fmt=".1f",
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            annot_kws={"size": fontsize * 0.7}  # 控制格子里数字大小
        )

        if show_xlabel:
            ax.set_xlabel('τ', fontsize=fontsize)
        if show_ylabel:
            ax.set_ylabel('k', fontsize=fontsize)

        ax.tick_params(axis='x', labelsize=fontsize * 0.8)
        ax.tick_params(axis='y', labelsize=fontsize * 0.8)
        ax.set_title(f'{sr}', fontsize=fontsize + 2)

        ax.invert_yaxis()

        if mappable is None:
            mappable = heatmap.get_children()[0]

    # 添加统一 colorbar
    cbar_ax = fig.add_axes([1.02, 0.05, 0.02, 0.915])
    cbar = fig.colorbar(mappable, cax=cbar_ax, label='Average Accuracy (%)')
    cbar.ax.tick_params(labelsize=fontsize * 0.8)
    cbar.ax.set_ylabel('Average Accuracy (%)', fontsize=fontsize)

    plt.show()

# %% Topographic plot
def topographic_plot(accuracy_data, lambda_values, sigma_values, max_mark=False, fontsize=12):
    LAMBDA, SIGMA = np.meshgrid(lambda_values, sigma_values, indexing='ij')
    vmin, vmax = 60, 95
    fig, axes = plt.subplots(1, 4, figsize=(15, 5), constrained_layout=True)
    axes = axes.flatten()
    sr_keys = list(accuracy_data.keys())
    
    contour_mappable = None
    
    for i, sr in enumerate(sr_keys):
        ax = axes[i]
        acc_matrix = accuracy_data[sr]
        Z = acc_matrix.astype(float)
    
        contour = ax.contourf(SIGMA, LAMBDA, Z, levels=20, cmap='coolwarm', vmin=vmin, vmax=vmax)
        ax.contour(SIGMA, LAMBDA, Z, levels=10, colors='k', linewidths=0.5, alpha=0.5)
        
        if True: # i % 4 == 0:
            ax.set_ylabel('k', fontsize=fontsize)
        else:
            ax.set_yticklabels([])
    
        if i >= 4:
            ax.set_xlabel('τ', fontsize=fontsize)
        else:
            ax.set_xticklabels([])
    
        ax.tick_params(axis='x', labelsize=fontsize * 0.8)
        ax.tick_params(axis='y', labelsize=fontsize * 0.8)
        ax.set_title(f'NRR, n={sr.split("=")[1]}', fontsize=fontsize + 2)
    
        if max_mark:
            max_idx = np.unravel_index(np.argmax(Z), Z.shape)
            ax.plot(SIGMA[max_idx], LAMBDA[max_idx], 'ro', color='green')
    
        if contour_mappable is None:
            contour_mappable = contour

    cbar_ax = fig.add_axes([1.02, 0.05, 0.02, 0.915])
    cbar = fig.colorbar(contour_mappable, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=fontsize * 0.8)
    cbar.ax.set_ylabel('Average Accuracy (%)', fontsize=fontsize)

    plt.show()


# %% Execute
# 原始 lambda 和 sigma
k_values = np.array([10, 20, 50, 100])
tau_values = np.array([0.2, 0.3, 0.5])

# accuracy_data 略，保持原样
accuracy_data_pcc = {
    "NRR=0.5": np.array([
        [91.32, 91.49, 90.88],
        [91.84, 91.50, 90.50],
        [92.04, 91.88, 91.15],
        [91.99, 91.58, 91.06],
    ]),
    "NRR=0.3": np.array([
        [85.30, 85.29, 86.13],
        [86.25, 87.09, 87.14],
        [86.96, 86.91, 87.26],
        [88.45, 88.41, 87.81],
    ]),
    "NRR=0.2": np.array([
        [74.74, 75.67, 75.85],
        [79.49, 78.76, 79.96],
        [81.39, 81.92, 79.71],
        [82.32, 82.75, 79.19],
    ]),
    "NRR=0.1": np.array([
        [59.48, 59.41, 59.00],
        [59.70, 62.06, 59.42],
        [60.03, 62.21, 62.45],
        [62.93, 64.94, 62.13],
    ])
}

matrix_plot(accuracy_data_pcc, k_values, tau_values, fontsize=20)
topographic_plot(accuracy_data_pcc, k_values, tau_values, fontsize=20)
topographic_plot(accuracy_data_pcc, k_values, tau_values, max_mark=True, fontsize=20)