# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 16:55:00 2026

@author: 18307
"""

import numpy as np

import feature_engineering
def feature_fussion_color_blocking(fn_basis, fn_modifier, params={'normalization': True, 'scale': (0,1)}):
    normalization = params.get('normalization', True)
    scale = params.get('scale', (0,1))
    
    # print('Normalization: ', normalization)
    
    if normalization:
        fn_basis = feature_engineering.normalize_matrix(fn_basis, 'minmax', param={'target_range': scale})
        fn_modifier = feature_engineering.normalize_matrix(fn_modifier, 'minmax', param={'target_range': scale})
    else: 
        fn_basis = np.array(fn_basis)
        fn_modifier = np.array(fn_modifier)
    
    upper = np.triu(fn_basis)
    lower = np.tril(fn_modifier)
    
    fn_fussed = upper + lower
    
    return fn_fussed

def feature_fussion_additive(fn_basis, fn_modifier):
    fn_fussed = fn_basis + fn_modifier
    
    return fn_fussed

def feature_fussion_multiplicative(fn_basis, fn_modifier):
    fn_fussed = fn_basis * fn_modifier
    
    return fn_fussed

def map_to_nearest(x, sequence=[0.01, 0.25, 0.5, 0.75, 1]):
    seq = np.asarray(sequence)
    x = np.asarray(x)

    if np.any((x < 0) | (x > 1)):
        raise ValueError("x 中的元素必须在 [0, 1] 区间内")

    # 扩展维度以便广播
    diff = np.abs(x[..., None] - seq)   # shape: (*x.shape, len(seq))
    indices = np.argmin(diff, axis=-1)
    return seq[indices]

def feature_fussion_power_gating(fn_basis, fn_modifier, params={'power': 1, 'normalization': True, 'scale': (0,1)}):
    power = params.get('power', 1)
    normalization = params.get('normalization', True)
    scale = params.get('scale', (0,1))
    
    if normalization:
        fn_basis = feature_engineering.normalize_matrix(fn_basis, 'minmax', param={'target_range': scale})
        fn_modifier = feature_engineering.normalize_matrix(fn_modifier, 'minmax', param={'target_range': scale})
    else: 
        fn_basis = np.array(fn_basis)
        fn_modifier = np.array(fn_modifier)
    
    alpha = map_to_nearest(fn_modifier**power)
    
    fn_fussed = fn_basis * alpha
    
    return fn_fussed

from utils import utils_feature_loading
def feature_fussion_power_gating_parameterization(fn_basis, params={'power': 1, 'normalization': True, 'scale': (0,1)}):
    power = params.get('power', 1)
    normalization = params.get('normalization', True)
    scale = params.get('scale', (0,1))
    
    fcs_global_averaged = utils_feature_loading.read_fcs_global_average('seed', 'plv', 'joint', range(1,6))
    alpha_global_averaged = fcs_global_averaged['alpha']
    beta_global_averaged = fcs_global_averaged['beta']
    gamma_global_averaged = fcs_global_averaged['gamma']
    
    fn_modifier = alpha_global_averaged + beta_global_averaged + gamma_global_averaged
    
    if normalization:
        fn_basis = feature_engineering.normalize_matrix(fn_basis, 'minmax', param={'target_range': scale})
        fn_modifier = feature_engineering.normalize_matrix(fn_modifier, 'minmax', param={'target_range': scale})
    else: 
        fn_basis = np.array(fn_basis)
        fn_modifier = np.array(fn_modifier)
    
    alpha = map_to_nearest(fn_modifier**power)
    
    fn_fussed = fn_basis * alpha
    
    return fn_fussed

def feature_fusion_sigmoid_gating(fn_basis, fn_modifier, params={'k': 10.0, # gate sharpness
                                                                 'tau': 0.5, # confidence threshold
                                                                 'discretization': False, 'sequence': [0.01, 0.25, 0.5, 0.75, 1],
                                                                 'normalization': True, 'scale': (0, 1)}):
    k = params.get('k', 10.0)
    tau = params.get('tau', 0.5)
    discretization = params.get('discretization', False)
    sequence = params.get('sequence', [0.01, 0.25, 0.5, 0.75, 1])
    normalization = params.get('normalization', True)
    scale = params.get('scale', (0, 1))

    if normalization:
        fn_basis = feature_engineering.normalize_matrix(fn_basis, 'minmax', param={'target_range': scale})
        fn_modifier = feature_engineering.normalize_matrix(fn_modifier, 'minmax', param={'target_range': scale})
    else:
        fn_basis = np.array(fn_basis)
        fn_modifier = np.array(fn_modifier)
        
    tau = np.quantile(fn_modifier, tau)    
    
    # sigmoid confidence gate
    alpha = 1.0 / (1.0 + np.exp(-k * (fn_modifier - tau)))
    
    if discretization:
        alpha = map_to_nearest(alpha, sequence)
    
    fn_fused = fn_basis * alpha
    return fn_fused

import cluster
def feature_fussion_cluster_based(fn_basis, fn_modifier, 
                                  dm_params={'source': 'auto', 'type': '3d_euclidean'},
                                  cluster_params={'threshold': None},
                                  fussion_params={'alpha': 0, 'beta': 0}):
    cluster_threshold = cluster_params.get('threshold', None)
    alpha, beta = fussion_params.get('alpha', 0), fussion_params.get('beta', 0)
    
    electrodes, dm = feature_engineering.compute_distance_matrix('seed', dm_params)
    clusters, parsed_clusters = cluster.hierarchical_clustering(dm, cluster_threshold, electrodes, parse=True, verbose=True)
    
    inter_cluster_mask = cluster.construct_inter_cluster_mask(clusters['cluster_idx'])
    
    fn, fn_basis, fn_modifier = cluster.reconstruct_fn(fn_basis, fn_modifier, inter_cluster_mask,
                                                       params={'alpha': alpha, 'beta': beta, 'scale': (0, 1)})
    
    return fn, fn_basis, fn_modifier

def feature_fussion(fns_1, fns_2, params={}):
    fussion_type = params.get('fussion_type').lower()
    
    fussion_type_valid = {'cluster', 'color_blocking', 'additive', 'multiplicative', 'power_gating', 'power_gating_parameterization', 'sigmoid_gating'}
    if fussion_type not in fussion_type_valid:
        raise ValueError(f"Invalid filter '{fussion_type}'. Allowed filters: {fussion_type_valid}")
    
    if fussion_type == 'cluster':
        fn_fussed, fn_basis, fn_modifier = feature_fussion_cluster_based(fns_1, fns_2, 
                                                                         params.get('dm_params'),
                                                                         params.get('cluster_params'),
                                                                         params.get('fussion_params'))
    elif fussion_type == 'additive':
        fn_fussed = feature_fussion_additive(fns_1, fns_2)
    elif fussion_type == 'multiplicative':
        fn_fussed = feature_fussion_multiplicative(fns_1, fns_2)
    elif fussion_type == 'color_blocking':
        fn_fussed = feature_fussion_color_blocking(fns_1, fns_2, params)
    elif fussion_type == 'power_gating':
        fn_fussed = feature_fussion_power_gating(fns_1, fns_2, params)
    elif fussion_type == 'power_gating_parameterization':
        fn_fussed = feature_fussion_power_gating_parameterization(fns_1, params)
    elif fussion_type == 'sigmoid_gating':
        fn_fussed = feature_fusion_sigmoid_gating(fns_1, fns_2, params)
    
    return fn_fussed

if __name__ == "__main__":
    from utils import utils_feature_loading
    from utils import utils_visualization
    
    feature_basis='pcc'
    feature_modifier='plv'
    params={'fussion_type': 'cluster',                                                
            'dm_params': {"source": "auto", "type": "3d_euclidean"},
            'cluster_params': {'threshold': None},
            'fussion_params': {'alpha': 0, 'beta': 0},}
    
    fcs_basis_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_basis, 'joint', range(1,6))
    alpha_basis_global_averaged = fcs_basis_global_averaged['alpha']
    beta_basis_global_averaged = fcs_basis_global_averaged['beta']
    gamma_basis_global_averaged = fcs_basis_global_averaged['gamma']
    
    fcs_modifier_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_modifier, 'joint', range(1,6))
    alpha_modifier_global_averaged = fcs_modifier_global_averaged['alpha']
    beta_modifier_global_averaged = fcs_modifier_global_averaged['beta']
    gamma_modifier_global_averaged = fcs_modifier_global_averaged['gamma']
    
    alpha_fussed_global_averaged = feature_fussion(alpha_basis_global_averaged, alpha_modifier_global_averaged, params)
    beta_fussed_global_averaged = feature_fussion(beta_basis_global_averaged, beta_modifier_global_averaged, params)
    gamma_fussed_global_averaged = feature_fussion(gamma_basis_global_averaged, gamma_modifier_global_averaged, params)
    
    utils_visualization.draw_projection(alpha_basis_global_averaged)
    utils_visualization.draw_projection(alpha_modifier_global_averaged)
    utils_visualization.draw_projection(alpha_fussed_global_averaged)
    
    #
    params = {'fussion_type': 'sum'}
    alpha_fussed = feature_fussion(alpha_basis_global_averaged, alpha_modifier_global_averaged, params)
    utils_visualization.draw_projection(alpha_fussed)
    
    params = {'fussion_type': 'color_blocking'}
    alpha_fussed = feature_fussion(alpha_basis_global_averaged, alpha_modifier_global_averaged, params)
    utils_visualization.draw_projection(alpha_fussed)