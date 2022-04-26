'''
Description: 
Author: JinShiyin
Email: shiyinjin@foxmail.com
Date: 2022-04-26 14:43:34
'''
import os
import numpy as np
from tqdm import tqdm

all_sequences = [
    # ############### val
    'MPI_HDM05',
    'HumanEva', 
    'SFU', 
    'MPI_mosh', 
    # ############### test
    'Transitions_mocap', 
    'SSM_synced',
    # ############### train
    'CMU', 
    'DanceDB', 
    'EKUT', 
    'Eyes_Japan_Dataset', 
    'BMLmovi', 
    'GRAB', 
    'SOMA', 
    'DFaust_67', 
    'BioMotionLab_NTroje', 
    'TCD_handMocap', 
    'TotalCapture', 
    'MPI_Limits',  
    'KIT', 
    'BMLhandball', 
    'ACCAD', 
    'HUMAN4D', 
]

# original 
# all_sequences = [
#     # ############### val
#     'MPI_HDM05',
#     'HumanEva', 
#     'SFU', 
#     'MPI_mosh', 
#     # ############### test
#     'Transitions_mocap', 
#     'SSM_synced',
#     # ############### train
#     'CMU', 
#     'MPI_Limits', 
#     'TotalCapture', 
#     'Eyes_Japan_Dataset', 
#     'KIT', 
#     'BioMotionLab_NTroje', 
#     'EKUT', 
#     'TCD_handMocap', 
#     'ACCAD'
# ]


def get_vname(ori_v_seq_name):
    v_id = ori_v_seq_name.split("_")[0]
    if v_id == "MPI":
        sub_v_id = ori_v_seq_name.split("_")[1]
        if sub_v_id == "HDM05":
            return "MPI_HDM05"
        elif sub_v_id == "mosh":
            return "MPI_mosh"
        elif sub_v_id == "Limits":
            return "MPI_Limits"
    elif v_id == "SSM":
        return "SSM_synced"
    elif v_id == "Transitions":
        return "Transitions_mocap"
    elif v_id == "Eyes":
        return "Eyes_Japan_Dataset"
    elif v_id == "TCD":
        return "TCD_handMocap"
    elif v_id == "DFaust":
        return "DFaust_67"
    elif v_id == "BioMotionLab":
        return "BioMotionLab_NTroje"
    else:
        return v_id

# def get_mean_and_std(npy_folder, output_path):
#     npy_files = os.listdir(npy_folder)
#     npy_files.sort()
#     data = None
#     for i in tqdm(range(len(npy_files)), ncols=100):
#         npy_name = npy_files[i]
#         vname = get_vname(npy_name)
#         if vname not in all_sequences:
#             print(f'{npy_name}, {vname} not in all_sequences, skip')
#             continue
#         npy_path = os.path.join(npy_folder, npy_name)
#         if data is None:
#             data = np.load(npy_path) # T X 579
#         else:
#             data = np.concatenate([data, np.load(npy_path)])
    
#     print(f'data.shape={data.shape}')
#     print(f'cal mean and std')
#     mean = np.mean(data, axis=0)
#     std = np.std(data, axis=0)
#     mean = np.expand_dims(mean, axis=0)
#     std = np.expand_dims(std, axis=0)
#     mean_std = np.concatenate([mean, std], axis=0)
#     np.save(output_path, mean_std)
#     print(mean_std)
#     print(mean_std.shape)
#     print(f'{mean_std} saved...')


def get_mean_and_std(npy_folder, output_path):
    npy_files = os.listdir(npy_folder)
    npy_files.sort()
    sum_for_mean = np.zeros((1, 579), dtype=np.float32) # 1 X 579
    cnt = 0
    for i in tqdm(range(len(npy_files)), ncols=100):
        npy_name = npy_files[i]
        vname = get_vname(npy_name)
        if vname not in all_sequences:
            # print(f'{npy_name}, {vname} not in all_sequences, skip')
            continue
        npy_path = os.path.join(npy_folder, npy_name)
        data = np.load(npy_path) # T X 579
        cnt += data.shape[0]
        data_sum = np.sum(data, axis=0, keepdims=True) # 1 X 579
        sum_for_mean += data_sum
    print(sum_for_mean)
    print(cnt)
    mean = sum_for_mean / cnt # 1 X 579

    sum_for_std = np.zeros((1, 579), dtype=np.float32) # 1 X 579
    for i in tqdm(range(len(npy_files)), ncols=100):
        npy_name = npy_files[i]
        vname = get_vname(npy_name)
        if vname not in all_sequences:
            # print(f'{npy_name}, {vname} not in all_sequences, skip')
            continue
        npy_path = os.path.join(npy_folder, npy_name)
        data = np.load(npy_path) # T X 579
        data -= mean
        data = np.square(data)
        data_sum = np.sum(data, axis=0, keepdims=True) # 1 X 579
        sum_for_std += data_sum
    print(sum_for_std)
    print(cnt)
    std = sum_for_std / cnt
    std = np.sqrt(std)
    mean_std = np.concatenate([mean, std], axis=0)
    np.save(output_path, mean_std)
    print(mean_std)
    print(mean_std.shape)
    print(f'{output_path} saved...')



if __name__ == '__main__':
    npy_folder = '/data/jsy/datasets/AMASS/amass_for_hm_vae_fps30'
    output_path = 'data/for_all_data_motion_model/all_amass_data_mean_std.npy'
    get_mean_and_std(npy_folder, output_path)
