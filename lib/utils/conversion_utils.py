'''
Description: 
Author: JinShiyin
Email: shiyinjin@foxmail.com
Date: 2022-04-22 21:49:45
'''
import torch
import numpy as np
import torchgeometry as tgm
from lib.utils.rotation_utils import get_base_rot_matrix, rotmat_to_rot6d


def get_amass_to_smpl_rot_mat():
    '''
    description: get rotation matrix of amass to smpl
    param {*}
    return {torch.FloatTensor} rot_mat: 3 X 3
    '''
    # The rotation matrix is orthogonal, so the inverse matrix is equal to the transpose of the matrix.
    rot_mat = get_smpl_to_amass_rot_mat() # 3 X 3
    rot_mat = rot_mat.transpose(1, 0) 
    return rot_mat

def get_smpl_to_amass_rot_mat():
    '''
    description: get rotation matrix of smpl to amass
    param {*}
    return {torch.FloatTensor} rot_mat: 3 X 3
    '''
    rot_mat1 = get_base_rot_matrix(angle='x', degree=90)
    rot_mat2 = get_base_rot_matrix(angle='y', degree=90)
    rot_mat = torch.matmul(rot_mat1, rot_mat2)
    return rot_mat

def amass_pose_to_smpl_pose(amass_rot_mat_pose):
    '''
    description: 
    param {torch.FloatTensor} amass_rot_mat_pose: T X n_joints X 3 X 3
    return {torch.FloatTensor} amass_rot_mat_pose: T X n_joints X 3 X 3
    '''
    device = amass_rot_mat_pose.device
    root_orientation = amass_rot_mat_pose[:, 0, :, :] # T X 3 X 3
    rot_mat_amass_to_smpl = get_amass_to_smpl_rot_mat() # 3 X 3
    rot_mat_amass_to_smpl = rot_mat_amass_to_smpl.to(device)
    rot_mat_amass_to_smpl = rot_mat_amass_to_smpl.unsqueeze(0) # 1 X 3 X 3
    root_orientation = torch.matmul(rot_mat_amass_to_smpl, root_orientation) # T X 3 X 3
    amass_rot_mat_pose[:, 0, :, :] = root_orientation
    return amass_rot_mat_pose


def smpl_pose_to_amass_pose(smpl_rot_mat_pose):
    '''
    description: 
    param {torch.FloatTensor} smpl_rot_mat_pose: T X n_joints X 3 X 3
    return {torch.FloatTensor} smpl_rot_mat_pose: T X n_joints X 3 X 3
    '''
    device = smpl_rot_mat_pose.device
    root_orientation = smpl_rot_mat_pose[:, 0, :, :] # T X 3 X 3
    rot_mat_smpl_to_amass = get_smpl_to_amass_rot_mat() # 3 X 3
    rot_mat_smpl_to_amass = rot_mat_smpl_to_amass.to(device)
    rot_mat_smpl_to_amass = rot_mat_smpl_to_amass.unsqueeze(0) # 1 X 3 X 3
    root_orientation = torch.matmul(rot_mat_smpl_to_amass, root_orientation) # T X 3 X 3
    smpl_rot_mat_pose[:, 0, :, :] = root_orientation
    return smpl_rot_mat_pose


def amass_verts_to_smpl_verts(amass_verts):
    '''
    description: 
    param {torch.FloatTensor} amass_verts: T X n_joints X 3, n_joints is usually set to 6980
    return {torch.FloatTensor} smpl_verts: T X n_joints X 3
    '''
    timesteps, n_joints, _ = amass_verts.size()
    device = amass_verts.device
    amass_verts = amass_verts.view(-1, 3).unsqueeze(2) # (T*n_joints) X 3 X 1
    rot_mat_amass_to_smpl = get_amass_to_smpl_rot_mat() # 3 X 3
    rot_mat_amass_to_smpl = rot_mat_amass_to_smpl.unsqueeze(0).to(device) # 1 X 3 X 3
    smpl_verts = torch.matmul(rot_mat_amass_to_smpl, amass_verts) # (T*n_joints) X 3 X 1
    smpl_verts = smpl_verts.squeeze(2).view(timesteps, n_joints, -1)
    return smpl_verts


def standardize_data_specify_dim(ori_data, mean_std_data, start_idx, end_idx):
    '''
    description: 
    param {*} self
    param {*} ori_data: bs X T X n_dim
    param {*} mean_std_data: 2 X 579
    param {*} start_idx
    param {*} end_idx
    return {*}
    '''
    # ori_data: T X n_dim
    mean_val = mean_std_data[[0], start_idx:end_idx].unsqueeze(0) # 1 X 1 X n_dim
    std_val = mean_std_data[[1], start_idx:end_idx].unsqueeze(0) # 1 X 1 X n_dim
    dest_data = (ori_data - mean_val)/(std_val+1e-9) # T X n_dim
    return dest_data


def convert_to_input(data, fk_layer, mean_std_data=None):
    '''
    description: 
    param {torch.tensor} data: bs X T X 144
    param {*} fk_layer
    param {torch.tensor} mean_std_data: 2 X 579, None->not normalize
    return {*}
    '''
    angle_axis = data[:, :, :72] # bs X T X (24*3)
    trans = data[:, :, 72:] # bs X T X 3
    bs, timesteps, _ = angle_axis.size()
    rot_mat = tgm.angle_axis_to_rotation_matrix(angle_axis.contiguous().view(-1, 3))[:, :3, :3].contiguous().view(-1, 24, 3, 3) # (bs*T) X 24 X 3 X 3
    rot_pos = fk_layer(rot_mat) # (bs*T) X 24 X 3
    rot_6d = rotmat_to_rot6d(rot_mat) # (bs*T) X 24 X 6
    
    # resize
    rot_6d = rot_6d.contiguous().view(bs, timesteps, -1) # bs X T X (24*6)
    rot_mat = rot_mat.contiguous().view(bs, timesteps, -1) # bs X T X (24*3*3)
    rot_pos = rot_pos.contiguous().view(bs, timesteps, -1) # bs X T X (24*3)

    # cal normilized_rot_pos
    # start_idx = 24*6+24*3*3+24*3
    # joint_pos = standardize_data_specify_dim(rot_pos, mean_std_data, start_idx, start_idx+24*3) # bs X T X (24*3)

    # cal linear_v
    minus_rot_pos = rot_pos[:, :-1, :] # bs X T-1 X (24*3)
    minus_rot_pos = torch.cat([rot_pos[:, [0], :], minus_rot_pos], dim=1) # bs X T X (24*3)
    linear_v = rot_pos - minus_rot_pos # bs X T X (24*3)
    if mean_std_data is not None:
        start_idx = 24*6 + 24*3*3 + 24*3
        linear_v = standardize_data_specify_dim(linear_v, mean_std_data, start_idx, start_idx+24*3) # bs X T X (24*3)

    # cal root_v
    minus_trans = trans[:, :-1, :] # bs X T-1 X 3
    minus_trans = torch.cat([trans[:, [0], :], minus_trans], dim=1) # bs X T X 3
    root_v = trans - minus_trans # bs X T X 3
    if mean_std_data is not None:
        root_v = standardize_data_specify_dim(root_v, mean_std_data, 579-3, 579)

    return rot_6d, rot_mat, rot_pos, trans, linear_v, root_v



if __name__ == "__main__":
    rot1 = get_smpl_to_amass_rot_mat()
    rot2 = get_amass_to_smpl_rot_mat()
    res = torch.matmul(rot2, rot1)
    print(res)
    
