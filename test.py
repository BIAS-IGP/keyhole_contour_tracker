
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 09:59:00 2025

@author: rpordzik
"""
import numpy as np
import scipy as sc
import pandas as pd
import os
import copy
import multiprocessing as mp
import time as benchmark
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

from tqdm.contrib.concurrent import process_map
from tqdm.contrib.concurrent import istarmap


def weighted_average(data, data_std):
    
    if ((data_std == 0) | np.isnan(data_std)).all():
        data_std[:] = 1
    
    mask = (~np.isnan(data)) * (~np.isnan(data_std)) * (data_std != 0)
    
    data = data[mask]
    data_std = data_std[mask]
    
    if len(data) == 1:
        mean_val = data[0]
        std_val = data_std[0]
    else:
        weights = 1 / (data_std**2)
        sum_weights = np.sum(weights)
        
        mean_val = np.sum(data * weights) / sum_weights
        
        std_val = np.sqrt(np.sum(weights * (data - mean_val)**2) / sum_weights)
            
    return mean_val, std_val

def weighted_average_axis(data_in, data_std_in, axis):
    
    data = copy.deepcopy(data_in)
    data_std = copy.deepcopy(data_std_in)
    
    if axis == 0:
        data = np.transpose(data_in)
        data_std = np.transpose(data_std_in)
            
    n_rows = data.shape[0]
    mean_vals = np.zeros((n_rows,))
    std_vals = np.zeros((n_rows,))
    
    for ii, row in enumerate(data):
        row_std = data_std[ii, :]
        if all((row_std == 0) | np.isnan(row_std)):
            row_std[:] = 1
        
        mask = (~np.isnan(row)) * (~np.isnan(row_std)) * (row_std != 0)
        
        row = row[mask]
        row_std = row_std[mask]
        
        if len(data) == 1:
            mean_vals[ii] = row[0]
            std_vals[ii] = row_std[0]
        else:
            weights = 1 / (row_std**2)
            sum_weights = np.sum(weights)
            
            mean_val = np.sum(row * weights) / sum_weights
            
            std_val = np.sqrt(np.sum(weights * (row - mean_val)**2) / sum_weights)
            
            mean_vals[ii] = mean_val
            std_vals[ii] = std_val
            
    return mean_vals, std_vals

def probe_weighted_average(values, depths, probe_depths, d_probe):
    results = np.zeros((len(probe_depths),))
    results_dev = np.zeros((len(probe_depths),))
    R = d_probe / 2
    dh = np.mean(np.diff(depths))
    
    for ii, probe_depth in enumerate(probe_depths):
        A = []
        result = []
        for jj, val in enumerate(values['mean']):
            if np.isnan(val):
                continue
            
            center_distance = np.abs(probe_depth - depths[jj] - dh / 2)
            d1 = center_distance - dh / 2
            d2 = center_distance + dh / 2
            
            if d1 > R:
                continue
                
            if d2 > R:
                d2 = R
            
            dA = R * (d2 * np.sqrt(1 - np.power(d2, 2) / np.power(R, 2)) + 
                      R * np.arcsin(d2 / R) - d1 * np.sqrt(1 - np.power(d1, 2) / np.power(R, 2)) - 
                      R * np.arcsin(d1 / R))
            A.append(dA)
            result.append(val)
           
        A = np.array(A)
        result = np.array(result)
        
        results[ii] = np.sum(A * result) / np.sum(A)
        results_dev[ii] = np.sqrt(np.sum(A * (result - results[ii])**2) / np.sum(A))
        
    results = {
        'depth': np.array(probe_depths),
        'mean': results,
        'dev': results_dev
        }
        
    results = pd.DataFrame(results)
        
    return results

def integrate_radius_ellipse(a, b, a_dev, b_dev, theta_range, N_points):
    theta_vec = np.linspace(-theta_range/2, theta_range/2, N_points)
    d_theta = np.mean(np.diff(theta_vec))
    
    radius_vec = np.zeros((N_points, ))
    literature_radius_vec = np.zeros((N_points, ))
    radius_dev_vec = np.zeros((N_points, ))
    
    for ii, theta in enumerate(theta_vec):
        # radius_vec[ii] = np.sqrt(a**2 * np.power(np.cos(theta),2) + b**2 * np.power(np.sin(theta),2))
        radius_vec[ii] = (a**2 * np.power(np.sin(theta),2) + b**2 * np.power(np.cos(theta),2))**(3/2) / (a*b)
        radius_dev_vec[ii] = np.abs(a_dev * a * np.power(np.cos(theta),2) / radius_vec[ii]) + np.abs(b_dev * b * np.power(np.sin(theta),2) / radius_vec[ii])
        
    radius, radius_dev = weighted_average(radius_vec, radius_dev_vec)

    return radius, radius_dev

def prepare_OCT_depths(depths, ref_time):
     a= 1
    

# def generate_curvatures_batch(keyhole_axes, depths, d_depth, theta_range, N_integral_points):
#     geometry_type = "paraboloid"
#     geometry_radii = np.zeros((2, 2, 2))

#     curvatures = []
    
#     axes = keyhole_axes.to_numpy(dtype=float)
#     depth = depths.iloc[:,1].to_numpy(dtype=float)
#     depth = np.expand_dims(depth, axis=1)
#     time = depths.iloc[:,0].to_numpy(dtype=float)
    
#     data = np.hstack((axes, depth));
#     check_vec = np.zeros((data.shape[0],))
    
#     start = benchmark.time()
#     for index, row in enumerate(data):
#         if not any(np.isnan(row)) and all(row[1:] > 0):
#             check_vec[index] = 1
            
#             geometry_radii[0, 0, 0] = row[1] / 2
#             geometry_radii[1, 0, 0] = row[2] / 2
            
#             geometry = {
#                 'depth': row[3] / 1000,
#                 'type': geometry_type,
#                 'radius': geometry_radii
#                 }
#             curvatures.append(generate_curvatures(geometry, d_depth, theta_range, N_integral_points))

#         else:
#             curvatures.append(-1)
#     print("1st benchmark ",benchmark.time() - start)
    
#     time = np.array(time)
    
#     curvatures = {
#         'time': time,
#         'curvatures': curvatures,
#         'checks': check_vec
#         }
    
#     start_2 = benchmark.time()
#     curvatures, curvature_stats = evaluate_curvature_data(curvatures)
#     print("2nd benchmark ", benchmark.time() - start_2)
    
#     return curvatures, curvature_stats




# def generate_curvatures(geometry, d_depth, theta_range, N_integral_points):
    
#     theta_range = np.deg2rad(theta_range)
    
#     depth = geometry['depth']
#     depth_dev = 0
#     # depth_dev = geometry['depth'][1]
   
#     ref_depths = np.arange(0, geometry['depth'] + d_depth / 2, d_depth)
    
#     if ref_depths[-1] > geometry['depth']:
#         ref_depths = np.delete(ref_depths, -1)
    
#     N_slices = len(ref_depths) 
    
#     curvature = np.zeros((N_slices, ))
#     curvature_dev = np.zeros((N_slices,))
#     if geometry['type'] == "cone":
        
#         aa_top = geometry['radius'][0, 0, 0]
#         aa_top_dev = geometry['radius'][0, 1, 0]
        
#         bb_top = geometry['radius'][1, 0, 0]
#         bb_top_dev = geometry['radius'][1, 1, 0]
        
#         aa_bottom = geometry['radius'][0, 0, 1]
#         aa_bottom_dev = geometry['radius'][0, 1, 1]
        
#         bb_bottom = geometry['radius'][1, 0, 1]
#         bb_bottom_dev = geometry['radius'][1, 1, 1]
        
        
#         for ii, dd in enumerate(ref_depths):
#             aa = (aa_top - aa_bottom) * dd / depth + aa_bottom
#             aa_dev = np.abs(aa_top_dev * dd / depth) + np.abs(-dd / depth + 1) * aa_bottom_dev + np.abs((aa_top - aa_bottom) * dd * depth_dev / depth)
            
#             bb = (bb_top - bb_bottom) * dd / depth + bb_bottom
#             bb_dev = np.abs(bb_top_dev * dd / depth) + np.abs(-dd / depth + 1) * bb_bottom_dev + np.abs((bb_top - bb_bottom) * dd * depth_dev / depth)
            
#             curvature[ii], curvature_dev[ii] = integrate_radius_ellipse(aa, bb, theta_range, N_integral_points)
       

#     elif geometry["type"] == "paraboloid":
#         aa_top = geometry['radius'][0, 0, 0]
#         aa_top_dev = geometry['radius'][0, 1, 0]
        
#         bb_top = geometry['radius'][1, 0, 0]
#         bb_top_dev = geometry['radius'][1, 1, 0]
        
#         for ii, dd in enumerate(ref_depths):
#             aa = aa_top * np.sqrt(1 - dd / depth)
#             aa_dev = np.abs(aa_top_dev *  np.sqrt(1 - dd / depth)) + np.abs(aa_top * dd * depth_dev / (2 * np.sqrt(depth - dd)))
            
#             bb = bb_top * np.sqrt(1 - dd / depth)
#             bb_dev = np.abs(bb_top_dev *  np.sqrt(1 - dd / depth)) + np.abs(bb_top * dd * depth_dev / (2 * np.sqrt(depth - dd))) 
            
#             curvature[ii], curvature_dev[ii] = integrate_radius_ellipse(aa, bb, aa_dev, bb_dev, theta_range, N_integral_points)
#     else:
#         print("Invalid geometry type!")
#         curvature = -1
#         curvature_dev = -1

#     curvature = {
#         'depth': ref_depths,
#         'curvature': curvature,
#         'curvature_dev': curvature_dev
#         }
    
#     curvature = pd.DataFrame(curvature)
#     # curvature = curvature.dropna()

#     return curvature
        
def _generate_geometry(row, d_depth, geometry_type, theta_range, N_integral_points):
    if not any(np.isnan(row)) and all(row[1:] > 0):
        geometry_radii = np.zeros((2, 2, 2))
        geometry_radii[0, 0, 0] = row[1] / 2
        geometry_radii[1, 0, 0] = row[2] / 2
        geometry = {
            'depth': row[3] / 1000,
            'type': geometry_type,
            'radius': geometry_radii
        }
        curvature = generate_curvatures(geometry, d_depth, theta_range, N_integral_points)
        return curvature, 1
    else:
        return -1, 0
    

def generate_curvatures(geometry, d_depth, theta_range, N_integral_points, n_jobs=1):
    theta_range = np.deg2rad(theta_range)
    depth = geometry['depth']
    depth_dev = 0  # Not currently used, but in place for extension

    ref_depths = np.arange(0, depth + d_depth / 2, d_depth)
    if ref_depths[-1] > depth:
        ref_depths = np.delete(ref_depths, -1)

    if geometry['type'] == "cone":
        aa_top = geometry['radius'][0, 0, 0]
        aa_top_dev = geometry['radius'][0, 1, 0]
        bb_top = geometry['radius'][1, 0, 0]
        bb_top_dev = geometry['radius'][1, 1, 0]
        aa_bottom = geometry['radius'][0, 0, 1]
        aa_bottom_dev = geometry['radius'][0, 1, 1]
        bb_bottom = geometry['radius'][1, 0, 1]
        bb_bottom_dev = geometry['radius'][1, 1, 1]

        def process_cone(dd):
            aa = (aa_top - aa_bottom) * dd / depth + aa_bottom
            aa_dev = (np.abs(aa_top_dev * dd / depth) +
                      np.abs(-dd / depth + 1) * aa_bottom_dev +
                      np.abs((aa_top - aa_bottom) * dd * depth_dev / depth))

            bb = (bb_top - bb_bottom) * dd / depth + bb_bottom
            bb_dev = (np.abs(bb_top_dev * dd / depth) +
                      np.abs(-dd / depth + 1) * bb_bottom_dev +
                      np.abs((bb_top - bb_bottom) * dd * depth_dev / depth))

            return integrate_radius_ellipse(aa, bb, theta_range, N_integral_points)

        results = Parallel(n_jobs=n_jobs, backend = "threading")(delayed(process_cone)(dd) for dd in ref_depths)

    elif geometry["type"] == "paraboloid":
        aa_top = geometry['radius'][0, 0, 0]
        aa_top_dev = geometry['radius'][0, 1, 0]
        bb_top = geometry['radius'][1, 0, 0]
        bb_top_dev = geometry['radius'][1, 1, 0]

        def process_paraboloid(dd):
            sqrt_term = np.sqrt(1 - dd / depth)
            aa = aa_top * sqrt_term
            aa_dev = (np.abs(aa_top_dev * sqrt_term) +
                      np.abs(aa_top * dd * depth_dev / (2 * np.sqrt(depth - dd))))

            bb = bb_top * sqrt_term
            bb_dev = (np.abs(bb_top_dev * sqrt_term) +
                      np.abs(bb_top * dd * depth_dev / (2 * np.sqrt(depth - dd))))

            return integrate_radius_ellipse(aa, bb, aa_dev, bb_dev, theta_range, N_integral_points)

        results = Parallel(n_jobs=n_jobs, backend = "threading")(delayed(process_paraboloid)(dd) for dd in ref_depths)

    else:
        print("Invalid geometry type!")
        return -1

    curvature, curvature_dev = zip(*results)

    return pd.DataFrame({
        'depth': ref_depths,
        'curvature': curvature,
        'curvature_dev': curvature_dev
    })

def generate_curvatures_batch(keyhole_axes, depths, d_depth, geometry_type, theta_range, N_integral_points):
    axes = keyhole_axes.to_numpy(dtype=float)
    depth = depths.iloc[:, 1].to_numpy(dtype=float).reshape(-1, 1)
    time = depths.iloc[:, 0].to_numpy(dtype=float)
    
    data = np.hstack((axes, depth))
    input_args = [(row, d_depth, geometry_type, theta_range, N_integral_points) for row in data]
    
    print("Processing...")
    bench1 = benchmark.time()
    with mp.Pool(mp.cpu_count()) as pool:
        results = list(
            istarmap(
                _generate_geometry,
                input_args,
                pool,
                total=len(input_args),
                desc="Generating geometries"
            )
        )
    print("1st benchmark ", benchmark.time() - bench1)
    curvatures, check_vec = zip(*results)
    check_vec = np.array(check_vec)
    curvatures = list(curvatures)

    curvature_dict = {
        'time': time,
        'curvatures': curvatures,
        'checks': check_vec
    }
    
    bench2 = benchmark.time()
    curvature_df, curvature_stats = evaluate_curvature_data(curvature_dict)
    print("2nd benchmark ", benchmark.time() - bench2)
    return curvature_df, curvature_stats

def calculate_probe_weighted_curvatures(curvature, probe_depths, probe_diameter):
    N_rows = curvature.size[0]
    
    depths = curvature['depth'].to_numpy()
    dh = np.mean(np.diff(depths))
    
    weighted_curvature = []
    
    for jj in range(len(probe_depths)):
        weighted_curvature.append({'values': 0, 'dA': 0})
    
    for index, row in curvature.iterrows():
        for jj in range(len(probe_depths)):
            center_distance = np.abs(row['depth'] - probe_depths[jj])
            val_temp, dA_temp = probe_weighted_average(row['curvature'], row['curvature_dev'], center_distance, dh, probe_diameter)
            weighted_curvature[jj]['values'] += val_temp
            weighted_curvature[jj]['dA'] += dA_temp
            
    for jj in range(len(probe_depths)):
        weighted_curvature[jj] = weighted_curvature[jj]['values'] / weighted_curvature[jj]['dA']
        
    weighted_curvatures = {
        'depth': probe_depths,
        'curvature': weighted_curvature,
        # 'curvature_dev': weighted_curvature_devs
        }
    
    return weighted_curvatures

def load_experimental_curvatures(file_path, experiment_name):
    # Excel-Datei einlesen
    file_path_curvatures = os.path.join(file_path, experiment_name + "_curvatures.csv")
    file_path_probe_weighted_curvatures = os.path.join(file_path, experiment_name + "_probe_weighted_curvatures.csv")
    
    curvatures = pd.read_csv(file_path_curvatures, sep=";")
    curvatures = curvatures.drop(curvatures.columns[-1], axis=1)
    
    probe_weighted_curvatures = pd.read_csv(file_path_probe_weighted_curvatures, sep=";")
    probe_weighted_curvatures = probe_weighted_curvatures.drop(probe_weighted_curvatures.columns[-1], axis=1)
    
    return curvatures, probe_weighted_curvatures

def load_keyhole_axes(file_path, time_range):
    # Excel-Datei einlesen   
    keyhole_axes = pd.read_csv(file_path, sep=";")
    
    time = keyhole_axes.iloc[:, 0].to_numpy(dtype=float)  
    indices = (time >= time_range[0]) * (time <= time_range[1])
    
    keyhole_axes = keyhole_axes.iloc[indices, :]
    
    temp = keyhole_axes.to_numpy(dtype=float)
    
    length_mean = np.nanmean(temp[:,1])
    length_dev = np.nanstd(temp[:,1])
    
    width_mean = np.nanmean(temp[:,2])
    width_dev = np.nanstd(temp[:,2])
    
    keyhole_mean_axes = {
        'length_mean': length_mean,
        'length_dev': length_dev,
        'width_mean': width_mean,
        'width_dev': width_dev
        }
    
    return keyhole_axes, keyhole_mean_axes

def load_oct_data(file_path):
    # Excel-Datei einlesen   
    oct_data = pd.read_csv(file_path, sep=";", header=35)
    oct_data = oct_data.drop(index=0)

    temp = oct_data.to_numpy(dtype=float)
    
    depth_mean = np.nanmean(temp[:,1])
    depth_dev = np.nanstd(temp[:,1])
    
    oct_depth_stats = {
        'depth_mean': depth_mean,
        'depth_dev': depth_dev
        }
    
    return oct_data, oct_depth_stats

def interpolate_oct_data(oct_data, ref_time):
    raw_data = oct_data.to_numpy(dtype=float)
    interpolator = sc.interpolate.interp1d(raw_data[:,0], raw_data[:,1], kind = 'linear')
    
    new_depth = interpolator(ref_time)
    
    result = {
        'time': ref_time,
        'OCT-depth': new_depth
        }
        
    result = pd.DataFrame(result)
    
    return result    

def aggregate_curvatures(curvature_stats, probe_depths, probe_diameter):
    curvatures_pw = curvature_stats['curvature']['time']

def evaluate_curvature_data(data):
    
    time_vec = data['time']
    time_len = len(time_vec)
    time = np.zeros((time_len, ))
    
    bench1 = benchmark.time()
    for ii, curve in enumerate(data['curvatures']):
        if type(curve) == int:
            continue
        depth_len_temp = curve.shape[0]
        if ii == 0:
            global_depth = curve['depth'].to_numpy(dtype=float)
            depth_len = depth_len_temp
        elif depth_len_temp > depth_len:
            global_depth = curve['depth'].to_numpy(dtype=float)
            depth_len = depth_len_temp
            
    print("3rd benchmark ", benchmark.time() - bench1)
    bench2 = benchmark.time()
    curvature_map = np.full((time_len, depth_len), np.nan)
    curvature_dev_map = np.full((time_len, depth_len), np.nan)
    
    # Build time and depth index maps once
    time_to_index = {t: i for i, t in enumerate(time_vec)}
    depth_to_index = {d: i for i, d in enumerate(global_depth)}
    
    for ii, curve in enumerate(data['curvatures']):
        if isinstance(curve, int):  # Skip invalid entries
            continue
    
        current_depth = curve['depth'].to_numpy(dtype=float)
        curvature = curve['curvature'].to_numpy(dtype=float)
        curvature_dev = curve['curvature_dev'].to_numpy(dtype=float)
    
        time_index = time_to_index.get(time_vec[ii], None)
        if time_index is None:
            continue  # Skip if time is not in vector (shouldn’t happen)
    
        try:
            depth_indices = [depth_to_index[dd] for dd in current_depth]
        except KeyError:
            continue  # Skip if any depth is not in global_depth (shouldn’t happen)
    
        curvature_map[time_index, depth_indices] = curvature
        curvature_dev_map[time_index, depth_indices] = curvature_dev
    print("last benchmark ", benchmark.time() - bench2)
    # Transpose to match original structure
    result = {
        'time': time_vec,
        'depth': global_depth,
        'checks': data['checks'],
        'curvature': curvature_map.T,
        'curvature_dev': curvature_dev_map.T
    }
    
    # Weighted averages
    curve_time, curve_time_dev = weighted_average_axis(curvature_map, curvature_dev_map, axis=0)
    curve_depth, curve_depth_dev = weighted_average_axis(curvature_map, curvature_dev_map, axis=1)
    curve_complete, curve_complete_dev = weighted_average(curvature_map, curvature_dev_map)
    
    result_stats = {
        'time': time_vec,
        'depth': global_depth,
        'curvature': {
            'time': {'mean': curve_time, 'dev': curve_time_dev},
            'depth': {'mean': curve_depth, 'dev': curve_depth_dev},
            'complete': {'mean': curve_complete, 'dev': curve_complete_dev}
        }
    }
    
    return result, result_stats

def write_curvatures(curvatures, curvature_stats, curvature_pw, output_folder, name):
    
    if not os.path.exists(output_folder):
       os.makedirs(output_folder)
    
    curve_out = np.full((len(curvatures['depth']) + 1, len(curvatures['time']) + 1), np.nan)
    curve_out[1:, 1:] = curvatures['curvature']
    curve_out[0, 1:] = curvatures['time']
    curve_out[1:, 0] = curvatures['depth']
    curve_out = pd.DataFrame(curve_out)
    
    curve_out.to_csv(os.path.join(output_folder, name + '_curvatures.csv'), sep = ';', index = False, header = False)
    curvature_pw.to_csv(os.path.join(output_folder, name + '_probe_weighted_curvatures.csv'), sep = ';', index = False)

def get_experiment_list(root_path, name_stem, video_config):
    evaluation_list = []
    video_subfolders = ['video_analysis', video_config]
    
    subdirs = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]
    subdirs = [directory for directory in subdirs if directory.startswith(name_stem)]
    
    for ii, current_dir in enumerate(subdirs):
        current_path = os.path.join(root_path, current_dir)
        
        name = current_dir
        
        oct_check = False
        axes_check = False
        
        for root, sub, files in os.walk(current_path):
            if files:
                file = [s for s in files if s.endswith('_keyhole_axes.csv')]
                if file and all(subfolder in root for subfolder in video_subfolders):
                    axes_check = True
                    axes_path = os.path.join(root, file[0])
                
                file = [s for s in files if s.endswith('_depth_analysis.oct')]
                if file:
                    oct_check = True
                    oct_path = os.path.join(root, file[0])
                    
        if oct_check and axes_check:
            temp = {
                'name': name,
                'oct_path': oct_path,
                'axes_path': axes_path
                }
            evaluation_list.append(temp)

    return evaluation_list

def main():
    time_range = [0.15, 0.6]
    theta_range = 120
    N_integral_points = 1000
    d_depth = 0.01
    
    probe_depths =  [1, 2, 3, 4, 5, 6]
    d_probe = 0.65
    
    geometry_type = "paraboloid"
    name_stem = 'IGP-H-V'
    video_config = 'config_I'
    sub_folders =  ['video_analyses']
    computer = "work"
    
    experiment_selection = []

    root_path = r"W:\IGP-H-V\Hauptversuche\Auswertungen_neu"        

    experiment_list = get_experiment_list(root_path, name_stem, video_config)
    
    for experiment in experiment_list:
        # try:
        name = experiment['name']
        axes_path = experiment['axes_path']
        oct_path = experiment['oct_path']
        
        if len(experiment_selection) > 0 and not name in experiment_selection:
            continue
        
        print(f"Curvature calculation for experiment {name} started...")
        
        output_folder = os.path.join(root_path, name, "curvature_calculation", video_config)
        
        keyhole_axes, keyhole_mean_axes = load_keyhole_axes(axes_path, time_range)
        oct_depth, oct_depth_stats = load_oct_data(oct_path)
        oct_depth = interpolate_oct_data(oct_depth, keyhole_axes.iloc[:,0])
         
        curvatures, curvature_stats = generate_curvatures_batch(keyhole_axes, oct_depth, d_depth, geometry_type, theta_range, N_integral_points)
        curvatures_pw = probe_weighted_average(curvature_stats['curvature']['time'], curvature_stats['depth'], probe_depths, d_probe)
        
        write_curvatures(curvatures, curvature_stats, curvatures_pw, output_folder, name)
        print(f"Curvature calculation for experiment {name} successful!")
        # except:
        #     print(f"Curvature calculation for experiment {name} failed!")
            
    #curvatures_pw = aggregate_curvatures(curvatures_stats, probe_depths, probe_diameter)
    
if __name__ == "__main__":
    main()