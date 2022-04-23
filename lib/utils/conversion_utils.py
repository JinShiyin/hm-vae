'''
Description: 
Author: JinShiyin
Email: shiyinjin@foxmail.com
Date: 2022-04-22 21:49:45
'''
import torch
import numpy as np
from lib.utils.rotation_utils import get_base_rot_matrix


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


if __name__ == "__main__":
    rot1 = get_smpl_to_amass_rot_mat()
    rot2 = get_amass_to_smpl_rot_mat()
    res = torch.matmul(rot2, rot1)
    print(res)
    
