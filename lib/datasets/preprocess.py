'''
Description: 
Author: JinShiyin
Email: shiyinjin@foxmail.com
Date: 2022-04-26 11:36:47
'''

import joblib
from tqdm import tqdm
import os.path as osp
import numpy as np
import argparse
import os
import sys
sys.path.append("../")


dict_keys = ['betas', 'dmpls', 'gender', 'mocap_framerate', 'poses', 'trans']

# extract SMPL joints from SMPL-H model
joints_to_use = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 37
])
joints_to_use = np.arange(0, 156).reshape((-1, 3))[joints_to_use].reshape(-1)

# all_sequences = [
#     'ACCAD',
#     'BioMotionLab_NTroje',
#     'CMU',
#     'EKUT',
#     'Eyes_Japan_Dataset',
#     'HumanEva',
#     'KIT',
#     'MPI_HDM05',
#     'MPI_Limits',
#     'MPI_mosh',
#     'SFU',
#     'SSM_synced',
#     'TCD_handMocap',
#     'TotalCapture',
#     'Transitions_mocap',
# ]

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
#     'DanceDB', 
#     'EKUT', 
#     'Eyes_Japan_Dataset', 
#     'BMLmovi', 
#     'GRAB', 
#     'SOMA', 
#     'DFaust_67', 
#     'BioMotionLab_NTroje', 
#     'TCD_handMocap', 
#     'TotalCapture', 
#     'MPI_Limits',  
#     'KIT', 
#     'BMLhandball', 
#     'ACCAD', 
#     'HUMAN4D', 
#     # 'WEIZMANN', 
#     # 'CNRS', 
# ]

amass_splits = {
    'val': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
    'test': ['Transitions_mocap', 'SSM_synced'],
    'train': ['CMU', 'DanceDB', 'EKUT', 'Eyes_Japan_Dataset', 'BMLmovi', 'GRAB', 'SOMA', 'DFaust_67', 'BioMotionLab_NTroje', 'TCD_handMocap', 'TotalCapture', 'MPI_Limits', 'KIT', 'BMLhandball', 'ACCAD','HUMAN4D',]
}

# walking_id = ["07", "08", "35", "36", "37", "38", "45", "46", "47", "78", "91"] # subject id
# indian_id = ["94"] # subject id
# salsa_id = ["60", "61"] # subject id


def read_data(folder, sequences, fps=30):
    # sequences = [osp.join(folder, x) for x in sorted(os.listdir(folder)) if osp.isdir(osp.join(folder, x))]

    # if sequences == 'all':
    #     sequences = all_sequences

    db = {
        'theta': [],
        'vid_name': [],
    }

    for seq_name in sequences:
        print(f'Reading {seq_name} sequence...')
        seq_folder = osp.join(folder, seq_name)

        thetas, vid_names = read_single_sequence(seq_folder, seq_name, fps=fps)
        # seq_name_list = np.array([seq_name]*thetas.shape[0])
        print(seq_name, 'number of videos', thetas.shape[0])
        db['theta'].append(thetas)
        db['vid_name'].append(vid_names)

    db['theta'] = np.concatenate(db['theta'], axis=0)
    db['vid_name'] = np.concatenate(db['vid_name'], axis=0)

    return db



def read_single_sequence(folder, seq_name, fps=30):
    # subjects = os.listdir(folder)
    ori_subjects = os.listdir(folder)
    subjects = []
    for subdir in ori_subjects:
        if not "DS_Store" in subdir and not '.txt' in subdir:
            subjects.append(subdir)

    thetas = []
    vid_names = []

    for subject in tqdm(subjects, ncols=100):
        actions = [x for x in os.listdir(osp.join(folder, subject)) if x.endswith('.npz')]

        for action in actions:
            fname = osp.join(folder, subject, action)
            
            if fname.endswith('shape.npz'):
                continue
                
            data = np.load(fname, allow_pickle=True)

            if 'mocap_framerate' in data:
                mocap_framerate = int(data['mocap_framerate'])
            elif 'mocap_frame_rate' in data:
                mocap_framerate = int(data['mocap_frame_rate'])
            else:
                print(
                    f'[mocap_framerate] and [mocap_frame_rate] are not in [{fname}], skip')
                continue

            if fps is not None:
                sampling_freq = mocap_framerate // fps
            else:
                sampling_freq = 1
            
            pose = data['poses'][0::sampling_freq, joints_to_use] # N X 72
            trans = data['trans'][0::sampling_freq] # N X 3

            if pose.shape[0] < 60:
                continue

            theta = np.concatenate([pose, trans], axis=1)
            vid_name = np.array([f'{seq_name}_{subject}_{action[:-4]}']*pose.shape[0])

            vid_names.append(vid_name)
            thetas.append(theta)

    return np.concatenate(thetas, axis=0), np.concatenate(vid_names, axis=0)


def read_seq_data(folder, nsubjects, fps):
    subjects = os.listdir(folder)
    sequences = {}

    assert nsubjects < len(subjects), 'nsubjects should be less than len(subjects)'

    for subject in subjects[:nsubjects]:
        actions = os.listdir(osp.join(folder, subject))

        for action in actions:
            data = np.load(osp.join(folder, subject, action))
            mocap_framerate = int(data['mocap_framerate'])
            sampling_freq = mocap_framerate // fps
            sequences[(subject, action)] = data['poses'][0::sampling_freq, joints_to_use]

    train_set = {}
    test_set = {}

    for i, (k,v) in enumerate(sequences.items()):
        if i < len(sequences.keys()) - len(sequences.keys()) // 4:
            train_set[k] = v
        else:
            test_set[k] = v

    return train_set, test_set

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='dataset directory', default='/data/jsy/datasets/AMASS/decompress')
    args = parser.parse_args()

    db = read_data(args.dir, sequences=amass_splits['train'])
    db_file = osp.join('/data/jsy/datasets/AMASS/db_lib', 'amass_train_db_fps30.pt')
    print(f'Saving AMASS dataset to {db_file}')
    joblib.dump(db, db_file)

    db = read_data(args.dir, sequences=amass_splits['val'])
    db_file = osp.join('/data/jsy/datasets/AMASS/db_lib', 'amass_val_db_fps30.pt')
    print(f'Saving AMASS dataset to {db_file}')
    joblib.dump(db, db_file)

    db = read_data(args.dir, sequences=amass_splits['test'])
    db_file = osp.join('/data/jsy/datasets/AMASS/db_lib', 'amass_test_db_fps30.pt')
    print(f'Saving AMASS dataset to {db_file}')
    joblib.dump(db, db_file)

