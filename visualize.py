'''
Description: 
Author: JinShiyin
Email: shiyinjin@foxmail.com
Date: 2022-04-21 23:03:26
'''
import torch
import joblib
import my_tools
from fk_layer import ForwardKinematicsLayer
import torchgeometry as tgm
import numpy as np
import utils.process_all_data_motion

from utils_common import show3Dpose_animation_multiple

def aa2matrot(pose):
    '''
    :param Nx1xnum_jointsx3
    :return: pose_matrot: Nxnum_jointsx3X3
    '''
    batch_size = pose.size(0)
    pose_body_matrot = tgm.angle_axis_to_rotation_matrix(pose.reshape(-1, 3))[:, :3, :3].contiguous().view(batch_size, 1, -1, 9)
    # bs X 1 X n_joints X 9
    pose_body_matrot = pose_body_matrot.view(batch_size, 1, 24, 3, 3) # bs X 1 X n_joints X 3 X 3
    pose_body_matrot = pose_body_matrot.squeeze(1) # bs X n_joints X 3 X 3
    return pose_body_matrot

def aa2others(aa_data):
    # aa_data: bs X T X 72
    # rest_skeleton_offsets: bs X T X 24 X 3
    bs, timesteps, _ = aa_data.size()
    aa_data = aa_data.view(bs*timesteps, 24, 3)[:, None, :, :] # (bs*T) X 1 X n_joints X 3
    rot_mat_data = aa2matrot(aa_data) # (bs*T) X n_joints X 3 X 3
    # Convert first timesteps's root rotation to Identity
    
    rotMatrices = rot_mat_data # (bs*T) X 24 X 3 X 3
    # Convert rotation matrix to 6D representation
    cont6DRep = torch.stack((rotMatrices[:, :, :, 0], rotMatrices[:, :, :, 1]), dim=-2) # (bs*T) X 24 X 2 X 3
    cont6DRep = cont6DRep.view(rotMatrices.size()[0], rotMatrices.size()[1], 6) # (bs*T) X 24 X 6

    cont6DRep = cont6DRep.view(bs, timesteps, -1) # bs X T X (24*6)
    rotMatrices = rotMatrices.view(bs, timesteps, -1) # bs X T X (24*3*3)

    return cont6DRep, rotMatrices


def vibe2amass_from_rotmat(rot_mat):
    '''
    description: 
    param {*} rot_mat: bs X T X 24 X 3 X 3
    return {*} rot_mat: bs X T X 24 X 3 X 3
    '''
    bs, timesteps, n_joints, _, _ = rot_mat.size()
    rot_mat = rot_mat.view(-1, 3, 3)
    transform_matrix1 = torch.tensor([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ]).float().cuda()
    transform_matrix1 = transform_matrix1.unsqueeze(0) # 1 X 3 X 3

    transform_matrix2 = torch.tensor([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ]).float().cuda()
    transform_matrix2 = transform_matrix2.unsqueeze(0) # 1 X 3 X 3
    transform_matrix = torch.bmm(transform_matrix2, transform_matrix1)
    print(f'transform_matrix.size={transform_matrix.size()}')

    rot_matrix = torch.matmul(transform_matrix, rot_mat) # (bs*timesteps*n_joints) X 3 X 3
    rot_matrix = rot_matrix.view(-1, n_joints, 3, 3) # (bs*timesteps) X n_joints X 3 X 3
    rot_6d = my_tools.rot_mat_to_6d(rot_matrix) # (bs*timesteps) X n_joints X 6
    rot_matrix = rot_matrix.view(bs, timesteps, n_joints, 3, 3)
    rot_6d = rot_6d.view(bs, timesteps, n_joints, 6)
    return rot_6d, rot_matrix

if __name__ == '__main__':

    fk_layer = ForwardKinematicsLayer()
    vibe_path = '/data/jsy/code/VIBE/output/sample_video/vibe_output.pkl'
    # vibe_path = '/data/jsy/code/VIBE/output/hiphop_clip1/vibe_output.pkl'
    vibe_pred_pkl = joblib.load(vibe_path)
    pred_theta_data = vibe_pred_pkl[1]['pose']
    pred_aa_data = torch.from_numpy(pred_theta_data).float().cuda()
    timesteps, _ = pred_aa_data.size()

    pred_aa_data = pred_aa_data[None, :, :] # 1 X T X 72
    pred_cont6DRep, pred_rot_mat = aa2others(pred_aa_data)
    # bs X T X (24*6), bs X T X (24*3*3)

    # aa_data = pred_aa_data.view(timesteps, 1, 24, 3)
    # rotMatrices = utils.process_all_data_motion.aa2matrot(aa_data)  # bs X 24 X 3 X 3
    #                 # Convert rotation matrix to 6D representation
    # pred_cont6DRep = torch.stack((rotMatrices[:, :, :, 0], rotMatrices[:, :, :, 1]), dim=-2)  # bs X 24 X 2 X 3
    # pred_cont6DRep = pred_cont6DRep.view(rotMatrices.size()[0], rotMatrices.size()[1], 6)  # bs X 24 X 6
    # pred_cont6DRep = pred_cont6DRep.view(1, timesteps, -1)

    bs, timesteps, _ = pred_cont6DRep.size()
    # transform from smpl to amass
    pred_rot_mat = pred_rot_mat.view(bs, timesteps, 24, 3, 3)
    pred_cont6DRep = pred_cont6DRep.view(bs, timesteps, 24, 6)
    root_rot_mat = pred_rot_mat[:, :, [0], :, :] # 1 X T X 1 X 3 X 3
    root_rot_6d, root_rot_mat = vibe2amass_from_rotmat(root_rot_mat)
    # 1 X T X 1 X 6, 1 X T X 1 X 3 X 3
    pred_cont6DRep[:, :, [0], :] = root_rot_6d.clone()
    root_rot_mat[:, :, [0], :, :] = root_rot_mat.clone()

    pred_cont6DRep = pred_cont6DRep.view(bs, timesteps, -1)
    pred_rot_mat = pred_rot_mat.view(bs, timesteps, -1)
    # end

    pred_6d_rot = pred_cont6DRep.view(bs, timesteps, 24, 6)
    pred_fk_pose = fk_layer(pred_6d_rot.squeeze(0)) # T X 24 X 3
    our_fk_pose = pred_fk_pose.clone()
    our_fk_pose[:, :, 0] += 1
    concat_seq_cmp = torch.cat((pred_fk_pose[None, :, :, :], our_fk_pose[None, :, :, :]), dim=0) # 2 X T X 24 X 3
    image_directory = 'outputs/test/test_smpl_to_amass'
    # Visualize single seq           
    print(f'start visualize...')
    show3Dpose_animation_multiple(concat_seq_cmp.data.cpu().numpy(), image_directory, 0, "cmp_vibe_ours_dance_vis", str(0), use_amass=True)


    ############################################
    # fk_layer = ForwardKinematicsLayer()
    # data = np.load(f'/data/jsy/datasets/AMASS/amass_for_hm_vae_fps30/CMU_79_79_31_poses.npy')
    # pred_cont6DRep = torch.from_numpy(data[:, :144]).float().cuda() # T X 24*6
    # timesteps, _ = pred_cont6DRep.size()
    # pred_6d_rot = pred_cont6DRep.view(timesteps, 24, 6)

    # pred_fk_pose = fk_layer(pred_6d_rot) # T X 24 X 3
    # our_fk_pose = pred_fk_pose.clone()
    # our_fk_pose[:, :, 0] += 1
    # concat_seq_cmp = torch.cat((pred_fk_pose[None, :, :, :], our_fk_pose[None, :, :, :]), dim=0) # 2 X T X 24 X 3
    # image_directory = 'outputs/test/test_smpl_to_amass'
    # # Visualize single seq           
    # print(f'start visualize...')
    # show3Dpose_animation_multiple(concat_seq_cmp.data.cpu().numpy(), image_directory, 0, "cmp_vibe_ours_dance_vis", str(1), use_amass=True)

