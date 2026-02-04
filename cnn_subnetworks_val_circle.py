# -*- coding: utf-8 -*-
"""
Created on Thu May 22 09:21:23 2025

@author: usouu
"""
import os
import numpy as np
import pandas as pd

import torch

import cnn_validation
from models import models
from utils import utils_feature_loading

# %% read parameters/save
def read_params(model='exponential', model_fm='basic', model_rcm='differ', folder='fitting_results(15_15_joint_band_from_mat)'):
    identifier = f'{model_fm.lower()}_fm_{model_rcm.lower()}_rcm'
    
    path_current = os.getcwd()
    path_fitting_results = os.path.join(path_current, 'fitting_results', folder)
    file_path = os.path.join(path_fitting_results, f'fitting_results({identifier}).xlsx')
    
    df = pd.read_excel(file_path).set_index('method')
    df_dict = df.to_dict(orient='index')
    
    model = model.upper()
    params = df_dict[model]
    
    return params

def save_to_xlsx_sheet(df, folder_name, file_name, sheet_name):
    output_dir = os.path.join(os.getcwd(), folder_name)
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)

    # if file exsist
    if os.path.exists(file_path):
        try:
            # try to read sheet
            existing_df = pd.read_excel(file_path, sheet_name=sheet_name)
        except ValueError:
            # if sheet not exsist then create empty DataFrame
            existing_df = pd.DataFrame()

        # concat by column
        df = pd.concat([existing_df, df], ignore_index=True)

        # continuation + replace
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    else:
        # if file not exsist then create
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    
def save_to_xlsx_fitting(results, subject_range, experiment_range, folder_name, file_name, sheet_name):
    # calculate average
    result_keys = results[0].keys()
    avg_results = {key: np.mean([res[key] for res in results]) for key in result_keys}
    
    # save to xlsx
    # 准备结果数据
    df_results = pd.DataFrame(results)
    df_results.insert(0, "Subject-Experiment", [f'sub{i}ex{j}' for i in subject_range for j in experiment_range])
    df_results.loc["Average"] = ["Average"] + list(avg_results.values())
    
    # 构造保存路径
    path_save = os.path.join(os.getcwd(), folder_name, file_name)
    
    # 判断文件是否存在
    if os.path.exists(path_save):
        # 追加模式，保留已有 sheet，添加新 sheet
        with pd.ExcelWriter(path_save, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df_results.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        # 新建文件
        with pd.ExcelWriter(path_save, engine='openpyxl') as writer:
            df_results.to_excel(writer, sheet_name=sheet_name, index=False)

# %% cnn subnetworks evaluation circle common
def cnn_subnetworks_evaluation_circle_original_cm(selection_rate=1, feature_cm='pcc', 
                                                 subject_range=range(6,16), experiment_range=range(1,4), 
                                                 subnetworks_extract='read', subnetworks_exrtact_basis=range(1,6),
                                                 save=False):
    if subnetworks_extract == 'read':
        fcs_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_cm, sub_range=subnetworks_exrtact_basis)
        alpha_global_averaged = fcs_global_averaged['alpha']
        beta_global_averaged = fcs_global_averaged['beta']
        gamma_global_averaged = fcs_global_averaged['gamma']
        
        strength_alpha = np.sum(np.abs(alpha_global_averaged), axis=1)
        strength_beta = np.sum(np.abs(beta_global_averaged), axis=1)
        strength_gamma = np.sum(np.abs(gamma_global_averaged), axis=1)
        
        channel_weights = {'alpha': strength_alpha, 
                           'beta': strength_beta,
                           'gamma': strength_gamma,
                           }

    elif subnetworks_extract == 'calculation':
        functional_node_strength = {'alpha': [], 'beta': [], 'gamma': []}
        
        for sub in subnetworks_exrtact_basis:
            for ex in experiment_range:
                subject_id = f"sub{sub}ex{ex}"
                print(f"Evaluating {subject_id}...")
                
                # CM/MAT
                # features = utils_feature_loading.read_fcs_mat('seed', subject_id, feature_cm)
                # alpha = features['alpha']
                # beta = features['beta']
                # gamma = features['gamma']
    
                # CM/H5
                features = utils_feature_loading.read_fcs('seed', subject_id, feature_cm)
                alpha = features['alpha']
                beta = features['beta']
                gamma = features['gamma']
                
                # Compute node strength
                strength_alpha = np.sum(np.abs(alpha), axis=1)
                strength_beta = np.sum(np.abs(beta), axis=1)
                strength_gamma = np.sum(np.abs(gamma), axis=1)
                
                # Save for further analysis
                functional_node_strength['alpha'].append(strength_alpha)
                functional_node_strength['beta'].append(strength_beta)
                functional_node_strength['gamma'].append(strength_gamma)
    
        channel_weights = {'gamma': np.mean(np.mean(functional_node_strength['gamma'], axis=0), axis=0),
                           'beta': np.mean(np.mean(functional_node_strength['beta'], axis=0), axis=0),
                           'alpha': np.mean(np.mean(functional_node_strength['alpha'], axis=0), axis=0)
                           }
    
    k = {'gamma': int(len(channel_weights['gamma']) * selection_rate),
         'beta': int(len(channel_weights['beta']) * selection_rate),
         'alpha': int(len(channel_weights['alpha']) * selection_rate),
          }
    
    channel_selects = {'gamma': np.argsort(channel_weights['gamma'])[-k['gamma']:][::-1],
                       'beta': np.argsort(channel_weights['beta'])[-k['beta']:][::-1],
                       'alpha': np.argsort(channel_weights['alpha'])[-k['alpha']:][::-1]
                       }
    
    # for traning and testing in CNN
    # labels
    labels = utils_feature_loading.read_labels(dataset='seed', header=True)
    y = torch.tensor(np.array(labels)).view(-1)
   
    # data and evaluation circle
    all_results_list = []
    for sub in subject_range:
        for ex in experiment_range:
            subject_id = f"sub{sub}ex{ex}"
            print(f"Evaluating {subject_id}...")
            
            # CM/MAT
            # features = utils_feature_loading.read_fcs_mat('seed', subject_id, feature_cm)
            # alpha = features['alpha']
            # beta = features['beta']
            # gamma = features['gamma']

            # CM/H5
            features = utils_feature_loading.read_fcs('seed', subject_id, feature_cm)
            alpha = features['alpha']
            beta = features['beta']
            gamma = features['gamma']

            # Selected CM           
            alpha_selected = alpha[:,channel_selects['alpha'],:][:,:,channel_selects['alpha']]
            beta_selected = beta[:,channel_selects['beta'],:][:,:,channel_selects['beta']]
            gamma_selected = gamma[:,channel_selects['gamma'],:][:,:,channel_selects['gamma']]
            
            x_selected = np.stack((alpha_selected, beta_selected, gamma_selected), axis=1)
            
            # cnn model
            cnn_model = models.CNN_2layers_adaptive_maxpool_3()
            # traning and testing
            result_RCM = cnn_validation.cnn_cross_validation(cnn_model, x_selected, y)
            
            # Flatten result and add identifier
            result_flat = {'Identifier': subject_id, **result_RCM}
            all_results_list.append(result_flat)
            
    # Convert list of dicts to DataFrame
    df_results = pd.DataFrame(all_results_list)
    
    # Compute mean of all numeric columns (excluding Identifier)
    mean_row = df_results.select_dtypes(include=[np.number]).mean().to_dict()
    mean_row['Identifier'] = 'Average'
    
    # Std
    std_row = df_results.select_dtypes(include=[np.number]).std(ddof=0).to_dict()
    std_row['Identifier'] = 'Std'
    
    df_results = pd.concat([df_results, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    
    # Save
    if save:        
        folder_name = 'results_cnn_subnetwork_evaluation'
        file_name = f'cnn_validation_SubRCM_{feature_cm}_origin.xlsx'
        sheet_name = f'sr_{selection_rate}'
        
        save_to_xlsx_sheet(df_results, folder_name, file_name, sheet_name)

    return df_results

import spatial_gaussian_smoothing
def cnn_subnetworks_evaluation_circle_common(projection_params={"source": "auto", "type": "3d_euclidean"},
                                             filtering_params={},
                                             feature_cm='pcc', 
                                             apply_filter='graph_laplacian_filtering',
                                             subject_range=range(6,16), experiment_range=range(1,4),
                                             subnetworks_extract='separate_index', selection_rate=1,
                                             subnets_exrtact_basis_sub=range(1,6), subnets_exrtact_basis_ex=range(1,4),
                                             save=False):
    # subnetworks selects;channel selects------start
    # valid filters
    filters_valid={'graph_laplacian_filtering', 'glf',
                   'graph_spectral_filtering', 'gsf',
                   'generalized_surface_laplacian_filtering', 'gslf'}
    if apply_filter not in filters_valid:
        raise ValueError(f"Invalid filter '{apply_filter}'. Allowed filters: {filters_valid}")
    
    # subnetwork extraction----start
    if subnetworks_extract == 'unify_index':
        fcs_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_cm, 'joint',
                                                                            subnets_exrtact_basis_sub)
        alpha_global_averaged = fcs_global_averaged['alpha']
        beta_global_averaged = fcs_global_averaged['beta']
        gamma_global_averaged = fcs_global_averaged['gamma']

    elif subnetworks_extract == 'separate_index':
        fcs_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_cm, 'joint',
                                                                            subnets_exrtact_basis_sub)
        alpha_global_averaged = fcs_global_averaged['alpha']
        beta_global_averaged = fcs_global_averaged['beta']
        gamma_global_averaged = fcs_global_averaged['gamma']
        
        alpha_global_averaged = spatial_gaussian_smoothing.fcs_filtering_common(alpha_global_averaged,
                                                                                projection_params, 
                                                                                filtering_params,
                                                                                apply_filter)
        beta_global_averaged = spatial_gaussian_smoothing.fcs_filtering_common(beta_global_averaged,
                                                                               projection_params, 
                                                                               filtering_params,
                                                                               apply_filter)
        gamma_global_averaged = spatial_gaussian_smoothing.fcs_filtering_common(gamma_global_averaged,
                                                                                projection_params, 
                                                                                filtering_params,
                                                                                apply_filter)
        
    strength_alpha = np.sum(np.abs(alpha_global_averaged), axis=0)
    strength_beta = np.sum(np.abs(beta_global_averaged), axis=0)
    strength_gamma = np.sum(np.abs(gamma_global_averaged), axis=0)
        
    channel_weights = {'gamma': strength_gamma, 'beta': strength_beta, 'alpha': strength_alpha}
        
    k = {'gamma': int(len(channel_weights['gamma']) * selection_rate),
         'beta': int(len(channel_weights['beta']) * selection_rate),
         'alpha': int(len(channel_weights['alpha']) * selection_rate),
          }
    
    channel_selects = {'gamma': np.argsort(channel_weights['gamma'])[-k['gamma']:][::-1],
                       'beta': np.argsort(channel_weights['beta'])[-k['beta']:][::-1],
                       'alpha': np.argsort(channel_weights['alpha'])[-k['alpha']:][::-1]
                       }
    # subnetworks selects;channel selects------end
    
    # for traning and testing in CNN------start
    # labels
    labels = utils_feature_loading.read_labels(dataset='seed', header=True)
    y = torch.tensor(np.array(labels)).view(-1)
    
    # data and evaluation circle
    all_results_list = []
    for sub in subject_range:
        for ex in experiment_range:
            subject_id = f"sub{sub}ex{ex}"
            print(f"Evaluating {subject_id}...")

            # CM/H5
            features = utils_feature_loading.read_fcs('seed', subject_id, feature_cm)
            alpha = features['alpha']
            beta = features['beta']
            gamma = features['gamma']
            
            # RCM
            alpha_rebuilded = spatial_gaussian_smoothing.fcs_filtering_common(alpha, 
                                                                              projection_params, filtering_params,
                                                                              apply_filter)
            beta_rebuilded = spatial_gaussian_smoothing.fcs_filtering_common(beta, 
                                                                             projection_params, filtering_params,
                                                                             apply_filter)
            gamma_rebuilded = spatial_gaussian_smoothing.fcs_filtering_common(gamma, 
                                                                              projection_params, filtering_params,
                                                                              apply_filter)
            
            alpha_rebuilded = alpha_rebuilded[:,channel_selects['alpha'],:][:,:,channel_selects['alpha']]
            beta_rebuilded = beta_rebuilded[:,channel_selects['beta'],:][:,:,channel_selects['beta']]
            gamma_rebuilded = gamma_rebuilded[:,channel_selects['gamma'],:][:,:,channel_selects['gamma']]
            
            x_rebuilded = np.stack((alpha_rebuilded, beta_rebuilded, gamma_rebuilded), axis=1)
            
            # cnn model
            cnn_model = models.CNN_2layers_adaptive_maxpool_3()
            # traning and testing
            result_RCM = cnn_validation.cnn_cross_validation(cnn_model, x_rebuilded, y)
            
            # Flatten result and add identifier
            result_flat = {'Identifier': subject_id, **result_RCM}
            all_results_list.append(result_flat)
            
    # Convert list of dicts to DataFrame
    df_results = pd.DataFrame(all_results_list)
    
    # Compute mean of all numeric columns (excluding Identifier)
    mean_row = df_results.select_dtypes(include=[np.number]).mean().to_dict()
    mean_row['Identifier'] = 'Average'
    
    # Std
    std_row = df_results.select_dtypes(include=[np.number]).std(ddof=0).to_dict()
    std_row['Identifier'] = 'Std'
    
    df_results = pd.concat([df_results, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    
    # Save
    if save:
        prj_type = projection_params.get('type')
        
        folder_name = 'results_cnn_subnetwork_evaluation'
        file_name = f'cnn_validation_{feature_cm}_by_{prj_type}_{apply_filter}_{subnetworks_extract}.xlsx'
        sheet_name = f'sr_{selection_rate}'
        
        save_to_xlsx_sheet(df_results, folder_name, file_name, sheet_name)
        
        # Save Summary (20251002)
        df_summary = pd.DataFrame([mean_row, std_row])
        save_to_xlsx_sheet(df_summary, folder_name, file_name, 'summary')
    
    return df_results

import feature_fussion
def cnn_subnetworks_evaluation_circle_feature_fussion(feature_basis='pcc', feature_modifier='plv', 
                                                      params={'fussion_type': 'cluster',
                                                              'dm_params': {"source": "auto", "type": "3d_euclidean"},
                                                              'cluster_params': {'threshold': None},
                                                              'fussion_params': {'alpha': 0, 'beta': 0},},
                                                      subject_range=range(6,16), experiment_range=range(1,4),
                                                      subnetworks_extract='separate_index', selection_rate=1,
                                                      subnets_exrtact_basis_sub=range(1,6), subnets_exrtact_basis_ex=range(1,4),
                                                      save=False):
    # subnetworks selects;channel selects------start
    # valid filters
    fussion_type = params.get('fussion_type', 'cluster')
    
    fussion_type_valid = {'cluster', 'color_blocking', 'additive', 'multiplicative', 'power_gating', 'power_gating_parameterization', 'sigmoid_gating'}
    if fussion_type not in fussion_type_valid:
        raise ValueError(f"Invalid filter '{fussion_type}'. Allowed filters: {fussion_type_valid}")
    
    # subnetwork extraction----start
    if subnetworks_extract == 'unify_index':
        fcs_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_basis, 'joint',
                                                                            subnets_exrtact_basis_sub)
        alpha_global_averaged = fcs_global_averaged['alpha']
        beta_global_averaged = fcs_global_averaged['beta']
        gamma_global_averaged = fcs_global_averaged['gamma']

    elif subnetworks_extract == 'separate_index':
        fcs_basis_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_basis, 'joint',
                                                                                  subnets_exrtact_basis_sub)
        alpha_basis_global_averaged = fcs_basis_global_averaged['alpha']
        beta_basis_global_averaged = fcs_basis_global_averaged['beta']
        gamma_basis_global_averaged = fcs_basis_global_averaged['gamma']
        
        fcs_modifier_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_modifier, 'joint',
                                                                                  subnets_exrtact_basis_sub)
        alpha_modifier_global_averaged = fcs_modifier_global_averaged['alpha']
        beta_modifier_global_averaged = fcs_modifier_global_averaged['beta']
        gamma_modifier_global_averaged = fcs_modifier_global_averaged['gamma']
        
        alpha_global_averaged = feature_fussion.feature_fussion(alpha_basis_global_averaged, alpha_modifier_global_averaged, params)
        beta_global_averaged = feature_fussion.feature_fussion(beta_basis_global_averaged, beta_modifier_global_averaged, params)
        gamma_global_averaged = feature_fussion.feature_fussion(gamma_basis_global_averaged, gamma_modifier_global_averaged, params)
        
    strength_alpha = np.sum(np.abs(alpha_global_averaged), axis=0)
    strength_beta = np.sum(np.abs(beta_global_averaged), axis=0)
    strength_gamma = np.sum(np.abs(gamma_global_averaged), axis=0)
        
    channel_weights = {'gamma': strength_gamma, 'beta': strength_beta, 'alpha': strength_alpha}
        
    k = {'gamma': int(len(channel_weights['gamma']) * selection_rate),
         'beta': int(len(channel_weights['beta']) * selection_rate),
         'alpha': int(len(channel_weights['alpha']) * selection_rate),
          }
    
    channel_selects = {'gamma': np.argsort(channel_weights['gamma'])[-k['gamma']:][::-1],
                       'beta': np.argsort(channel_weights['beta'])[-k['beta']:][::-1],
                       'alpha': np.argsort(channel_weights['alpha'])[-k['alpha']:][::-1]
                       }
    # subnetworks selects;channel selects------end
    
    # for traning and testing in CNN------start
    # labels
    labels = utils_feature_loading.read_labels(dataset='seed', header=True)
    y = torch.tensor(np.array(labels)).view(-1)
    
    # data and evaluation circle
    all_results_list = []
    for sub in subject_range:
        for ex in experiment_range:
            subject_id = f"sub{sub}ex{ex}"
            print(f"Evaluating {subject_id}...")

            # FN/H5
            features_basis = utils_feature_loading.read_fcs('seed', subject_id, feature_basis)
            alpha_basis = features_basis['alpha']
            beta_basis = features_basis['beta']
            gamma_basis = features_basis['gamma']
            
            features_modifier = utils_feature_loading.read_fcs('seed', subject_id, feature_modifier)
            alpha_modifier = features_modifier['alpha']
            beta_modifier = features_modifier['beta']
            gamma_modifier = features_modifier['gamma']
            
            # fussed FN
            alpha_fussed = feature_fussion.feature_fussion(alpha_basis, alpha_modifier, params)
            beta_fussed = feature_fussion.feature_fussion(beta_basis, beta_modifier, params)
            gamma_fussed = feature_fussion.feature_fussion(gamma_basis, gamma_modifier, params)
            
            alpha_fussed = alpha_fussed[:,channel_selects['alpha'],:][:,:,channel_selects['alpha']]
            beta_fussed = beta_fussed[:,channel_selects['beta'],:][:,:,channel_selects['beta']]
            gamma_fussed = gamma_fussed[:,channel_selects['gamma'],:][:,:,channel_selects['gamma']]
            
            x_rebuilded = np.stack((alpha_fussed, beta_fussed, gamma_fussed), axis=1)
            
            # cnn model
            cnn_model = models.CNN_2layers_adaptive_maxpool_3()
            # traning and testing
            result_RCM = cnn_validation.cnn_cross_validation(cnn_model, x_rebuilded, y)
            
            # Flatten result and add identifier
            result_flat = {'Identifier': subject_id, **result_RCM}
            all_results_list.append(result_flat)
            
    # Convert list of dicts to DataFrame
    df_results = pd.DataFrame(all_results_list)
    
    # Compute mean of all numeric columns (excluding Identifier)
    mean_row = df_results.select_dtypes(include=[np.number]).mean().to_dict()
    mean_row['Identifier'] = 'Average'
    
    # Std
    std_row = df_results.select_dtypes(include=[np.number]).std(ddof=0).to_dict()
    std_row['Identifier'] = 'Std'
    
    df_results = pd.concat([df_results, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    
    # Save
    if save:
        folder_name = 'results_cnn_subnetwork_evaluation'
        # file_name = f'cnn_validation_{params.get('fussion_type', '')}.xlsx'
        file_name = f'cnn_validation_{params.get('fussion_type', '')}_{params.get('k', '')}_{params.get('tau', '')}.xlsx'
        sheet_name = f'sr_{selection_rate}'
        
        save_to_xlsx_sheet(df_results, folder_name, file_name, sheet_name)
        
        # Save Summary (20251002)
        df_summary = pd.DataFrame([mean_row, std_row])
        save_to_xlsx_sheet(df_summary, folder_name, file_name, 'summary')
    
    return df_results

# %% end program
import time
import threading
def shutdown_with_countdown(countdown_seconds=30):
    """
    Initiates a shutdown countdown, allowing the user to cancel shutdown within the given time.

    Args:
        countdown_seconds (int): The number of seconds to wait before shutting down.
    """
    def cancel_shutdown():
        nonlocal shutdown_flag
        user_input = input("\nPress 'c' and Enter to cancel shutdown: ").strip().lower()
        if user_input == 'c':
            shutdown_flag = False
            print("Shutdown cancelled.")

    # Flag to determine whether to proceed with shutdown
    shutdown_flag = True

    # Start a thread to listen for user input
    input_thread = threading.Thread(target=cancel_shutdown, daemon=True)
    input_thread.start()

    # Countdown timer
    print(f"Shutdown scheduled in {countdown_seconds} seconds. Press 'c' to cancel.")
    for i in range(countdown_seconds, 0, -1):
        print(f"Time remaining: {i} seconds", end="\r")
        time.sleep(1)

    # Check the flag after countdown
    if shutdown_flag:
        print("\nShutdown proceeding...")
        os.system("shutdown /s /t 1")  # Execute shutdown command
    else:
        print("\nShutdown aborted.")

def end_program_actions(play_sound=True, shutdown=False, countdown_seconds=120):
    """
    Performs actions at the end of the program, such as playing a sound or shutting down the system.

    Args:
        play_sound (bool): If True, plays a notification sound.
        shutdown (bool): If True, initiates shutdown with a countdown.
        countdown_seconds (int): Countdown time for shutdown confirmation.
    """
    if play_sound:
        try:
            import winsound
            print("Playing notification sound...")
            winsound.Beep(1000, 500)  # Frequency: 1000Hz, Duration: 500ms
        except ImportError:
            print("winsound module not available. Skipping sound playback.")

    if shutdown:
        shutdown_with_countdown(countdown_seconds)

# %% Execute
def normal_evaluation_framework():
    # feature
    feature = 'pcc'
    
    # selection rates
    selection_rate_list = [1.0, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05]
    for selection_rate in selection_rate_list:
        #-----------------------------------------------------------------------
        # # original functional connectivity networks
        # cnn_subnetworks_evaluation_circle_original_cm(selection_rate=selection_rate, feature_cm=feature, 
        #                                               subject_range=range(6,16), experiment_range=range(1,4), 
        #                                               subnetworks_extract='read', subnetworks_exrtact_basis=range(1,6),
        #                                               save=True)
        
        #-----------------------------------------------------------------------
        # # graph_laplacian_filtering; highpass; 'alpha': 1; 'sigma': 'knn_median', 'k': 3
        # cnn_subnetworks_evaluation_circle_common(projection_params={"source": "auto", "type": "3d_euclidean",
        #                                                             }, 
        #                                          # '3d_euclidean', '3d_spherical'
        #                                          # 'graph_shortest_path', 'resistance_distance'
        #                                          filtering_params={'sigma': 'knn_median', 'k': 3,
        #                                                            'kernel_normalization': False,
        #                                                            'laplacian_normalization': False,
        #                                                            'alpha': 1, 'mode': 'highpass',
        #                                                            'lateral_mode': 'unilateral'}, # 'bilateral', 'unilateral'
        #                                          feature_cm=feature,
        #                                          apply_filter='graph_laplacian_filtering',
        #                                          subject_range=range(6,16), experiment_range=range(1,4),
        #                                          subnetworks_extract='separate_index', selection_rate=selection_rate,
        #                                          save=True)
        
        # cnn_subnetworks_evaluation_circle_common(projection_params={"source": "auto", "type": "graph_shortest_path",
        #                                                             # "graph": {'k': 3}
        #                                                             }, 
        #                                          # '3d_euclidean', '3d_spherical'
        #                                          # 'graph_shortest_path', 'resistance_distance'
        #                                          filtering_params={'sigma': 'knn_median', 'k': 3,
        #                                                            'kernel_normalization': False,
        #                                                            'laplacian_normalization': False,
        #                                                            'alpha': 1, 'mode': 'highpass',
        #                                                            'lateral_mode': 'unilateral'}, # 'bilateral', 'unilateral'
        #                                          feature_cm=feature,
        #                                          apply_filter='graph_laplacian_filtering',
        #                                          subject_range=range(6,16), experiment_range=range(1,4),
        #                                          subnetworks_extract='separate_index', selection_rate=selection_rate,
        #                                          save=True)
        
        # ----------------------------------------------------------------------
        cnn_subnetworks_evaluation_circle_feature_fussion(feature_basis='plv', feature_modifier='pcc', 
                                                          # feature_basis='pcc', feature_modifier='plv', 
                                                          params={'fussion_type': 'cluster',
                                                                  'dm_params': {"source": "auto", "type": "3d_euclidean"},
                                                                  'cluster_params': {'threshold': None},
                                                                  'fussion_params': {'alpha': 0, 'beta': 0},},
                                                          subject_range=range(6,16), experiment_range=range(1,4),
                                                          subnetworks_extract='separate_index', selection_rate=selection_rate,
                                                          subnets_exrtact_basis_sub=range(1,6), subnets_exrtact_basis_ex=range(1,4),
                                                          save=True)
        
        # cnn_subnetworks_evaluation_circle_feature_fussion(feature_basis='pcc', feature_modifier='plv', 
        #                                                   params={'fussion_type': 'power_gating', 
        #                                                           'power': 1,
        #                                                           'normalization': True},
        #                                                   subject_range=range(6,16), experiment_range=range(1,4),
        #                                                   subnetworks_extract='separate_index', selection_rate=selection_rate,
        #                                                   subnets_exrtact_basis_sub=range(1,6), subnets_exrtact_basis_ex=range(1,4),
        #                                                   save=True)
        
        # cnn_subnetworks_evaluation_circle_feature_fussion(feature_basis='pcc', feature_modifier='plv', 
        #                                                   params={'fussion_type': 'sigmoid_gating', 
        #                                                           'k': 20, 'tau': 0.2,
        #                                                           'discretization': False, 
        #                                                           'sequence': [0.01, 0.25, 0.5, 0.75, 1],
        #                                                           'normalization': True},
        #                                                   subject_range=range(6,16), experiment_range=range(1,4),
        #                                                   subnetworks_extract='separate_index', selection_rate=selection_rate,
        #                                                   subnets_exrtact_basis_sub=range(1,6), subnets_exrtact_basis_ex=range(1,4),
        #                                                   save=True)
        
        # cnn_subnetworks_evaluation_circle_feature_fussion(feature_basis='pcc', feature_modifier='plv', 
        #                                                   params={'fussion_type': 'additive'},
        #                                                   subject_range=range(6,16), experiment_range=range(1,4),
        #                                                   subnetworks_extract='unify_index', selection_rate=selection_rate,
        #                                                   subnets_exrtact_basis_sub=range(1,6), subnets_exrtact_basis_ex=range(1,4),
        #                                                   save=True)
        
        # cnn_subnetworks_evaluation_circle_feature_fussion(feature_basis='pcc', feature_modifier='plv', 
        #                                                   params={'fussion_type': 'multiplicative'},
        #                                                   subject_range=range(6,16), experiment_range=range(1,4),
        #                                                   subnetworks_extract='unify_index', selection_rate=selection_rate,
        #                                                   subnets_exrtact_basis_sub=range(1,6), subnets_exrtact_basis_ex=range(1,4),
        #                                                   save=True)
        
        # cnn_subnetworks_evaluation_circle_feature_fussion(feature_basis='pcc', feature_modifier='plv',
        #                                                   params={'fussion_type': 'color_blocking', 'normalization': False},
        #                                                   subject_range=range(6,16), experiment_range=range(1,4),
        #                                                   subnetworks_extract='unify_index', selection_rate=selection_rate,
        #                                                   subnets_exrtact_basis_sub=range(1,6), subnets_exrtact_basis_ex=range(1,4),
        #                                                   save=True)
        
if __name__ == '__main__':
    normal_evaluation_framework()
    
    # %% End
    end_program_actions(play_sound=True, shutdown=True, countdown_seconds=120)