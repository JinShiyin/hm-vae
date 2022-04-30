import os 
import numpy as np 
import json 
import random 
from tqdm import tqdm

# original hm-vae
# amass_splits = {
#     'vald': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
#     'test': ['Transitions_mocap', 'SSM_synced'],
#     'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BioMotionLab_NTroje', 'EKUT', 'TCD_handMocap', 'ACCAD']
# }

amass_splits = {
    'vald': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
    'test': ['Transitions_mocap', 'SSM_synced'],
    'train': ['CMU', 'DanceDB', 'EKUT', 'Eyes_Japan_Dataset', 'BMLmovi', 'GRAB', 'SOMA', 'DFaust_67', 'BioMotionLab_NTroje', 'TCD_handMocap', 'TotalCapture', 'MPI_Limits', 'KIT', 'BMLhandball', 'ACCAD','HUMAN4D',]
}

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

def gen_all_json(train_json, val_json, test_json, npy_folder):
    npy_files = os.listdir(npy_folder)

    train_dict = {}
    val_dict = {}
    test_dict = {}
    train_cnt = 0
    val_cnt = 0
    test_cnt = 0

    for f_name in tqdm(npy_files, ncols=100):
        frame_num = np.load(os.path.join(npy_folder, f_name)).shape[0]
        if get_vname(f_name) in amass_splits['train']:
            train_dict[train_cnt] = {
                'file_name': f_name,
                'frame_num': frame_num
            } 
            train_cnt += 1 
        if get_vname(f_name) in amass_splits['vald']:
            val_dict[val_cnt] = {
                'file_name': f_name,
                'frame_num': frame_num
            } 
            val_cnt += 1
        if get_vname(f_name) in amass_splits['test']:
            test_dict[test_cnt] = {
                'file_name': f_name,
                'frame_num': frame_num
            } 
            test_cnt += 1

    print("Training sequences:{0}".format(train_cnt))
    print("Validation sequences:{0}".format(val_cnt))
    print("Test sequences:{0}".format(test_cnt))

    json.dump(train_dict, open(train_json, 'w'))
    json.dump(val_dict, open(val_json, 'w'))
    json.dump(test_dict, open(test_json, 'w'))

def gen_json(train_json, val_json, npy_folder):  
    npy_files = os.listdir(npy_folder)
    num_seq = len(npy_files)
    train_num = int(num_seq*0.85)
    train_list = random.sample(npy_files, train_num)

    train_dict = {}
    val_dict = {}
    train_cnt = 0
    val_cnt = 0
    for f_name in npy_files:
        if f_name in train_list:
            train_dict[train_cnt] = f_name
            train_cnt += 1
        else:
            val_dict[val_cnt] = f_name 
            val_cnt += 1

    print("Training sequences:{0}".format(train_cnt))
    print("Validation sequences:{0}".format(val_cnt))

    json.dump(train_dict, open(train_json, 'w'))
    json.dump(val_dict, open(val_json, 'w'))

if __name__ == "__main__":
    # data_folder = "for_subset_motion_model"
    # if not os.path.exists(data_folder):
    #     os.makedirs(data_folder)
    # train_json = os.path.join(data_folder, "train_walk_subset_motion_data.json")
    # val_json = os.path.join(data_folder, "val_walk_subset_motion_data.json")
    # npy_folder = "./data/processed_walk_subset_data"
    # # gen_json(train_json, val_json, npy_folder)
    # # train cnt: 160, val cnt: 29

    # data_folder = "for_subset_motion_model"
    # if not os.path.exists(data_folder):
    #     os.makedirs(data_folder)
    # train_json = os.path.join(data_folder, "train_salsa_subset_motion_data.json")
    # val_json = os.path.join(data_folder, "val_salsa_subset_motion_data.json")
    # npy_folder = "./data/processed_salsa_subset_data"
    # # gen_json(train_json, val_json, npy_folder)
    # # train cnt: 18, val cnt: 4

    data_folder = "data/for_all_data_motion_model/train_val_test_json_fps30"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    train_json = os.path.join(data_folder, "train_all_amass_motion_data.json")
    val_json = os.path.join(data_folder, "val_all_amass_motion_data.json")
    test_json = os.path.join(data_folder, "test_all_amass_motion_data.json")
    npy_folder = "/data/jsy/datasets/AMASS/amass_for_hm_vae_fps30"
    gen_all_json(train_json, val_json, test_json, npy_folder)
    
    # Training sequences:10818
    # Validation sequences:363
    # Test sequences:140

    # Training sequences:14381
    # Validation sequences:363
    # Test sequences:140