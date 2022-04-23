'''
Description: 
Author: JinShiyin
Email: shiyinjin@foxmail.com
Date: 2022-04-23 20:20:55
'''
'''
Description: 
Author: JinShiyin
Email: shiyinjin@foxmail.com
Date: 2022-04-21 23:03:26
'''
import torch
import joblib
import torchgeometry as tgm
from fk_layer import ForwardKinematicsLayer
import torchgeometry as tgm
import numpy as np
from lib.utils.conversion_utils import amass_pose_to_smpl_pose, smpl_pose_to_amass_pose
from lib.utils.rotation_utils import hmvae_rot6d_to_rotmat, rotmat_to_rot6d
import utils.process_all_data_motion

from utils_common import show3Dpose_animation_multiple

if __name__ == '__main__':

    fk_layer = ForwardKinematicsLayer()
    vibe_path = '/data/jsy/code/VIBE/output/sample_video/vibe_output.pkl'
    # vibe_path = '/data/jsy/code/VIBE/output/hiphop_clip1/vibe_output.pkl'
    vibe_pred_pkl = joblib.load(vibe_path)
    pred_theta_data = vibe_pred_pkl[1]['pose'][0:16, :]

    pred_aa_data = torch.from_numpy(pred_theta_data).float().cuda() # T X 72
    pred_aa_data = pred_aa_data # T X 72
    timesteps, _ = pred_aa_data.size()
    pred_rot_mat = tgm.angle_axis_to_rotation_matrix(pred_aa_data.view(-1, 3))[:, :3, :3] # (T*24) X 3 X 3
    pred_rot_mat = pred_rot_mat.view(-1, 24, 3, 3) # T X 24 X 3 X 3
    pred_rot_mat = smpl_pose_to_amass_pose(pred_rot_mat) # T X 24 X 3 X 3

    pred_rot_6d = rotmat_to_rot6d(pred_rot_mat) # (T*24) X 6
    # pred_rot_mat = hmvae_rot6d_to_rotmat(pred_rot_6d) # (T*24) X 6
    # pred_rot_6d = rotmat_to_rot6d(pred_rot_mat) # (T*24) X 6
    pred_rot_6d = pred_rot_6d.view(-1, 24, 6) # T X 24 X 6
    pred_fk_pose = fk_layer(pred_rot_6d) # T X 24 X 3
    
    # show single
    # image_directory = 'outputs/test/test_smpl_to_amass'
    # # Visualize single seq           
    # print(f'start visualize...')
    # show3Dpose_animation_multiple(pred_fk_pose.unsqueeze(0).data.cpu().numpy(), image_directory, 0, "cmp_vibe_ours_dance_vis", str(5), use_amass=True)



    inv_rot_mat = hmvae_rot6d_to_rotmat(pred_rot_6d)
    inv_rot_mat = inv_rot_mat.view(-1, 24, 3, 3)
    inv_rot_mat = amass_pose_to_smpl_pose(inv_rot_mat)
    inv_rot_6d = rotmat_to_rot6d(inv_rot_mat)
    inv_rot_6d = inv_rot_6d.view(-1, 24, 6)
    inv_fk_pose = fk_layer(inv_rot_6d)
    inv_fk_pose[:, :, 0] += 1

    # our_fk_pose = pred_fk_pose.clone()
    # our_fk_pose[:, :, 0] += 1
    concat_seq_cmp = torch.cat((pred_fk_pose[None, :, :, :], inv_fk_pose[None, :, :, :]), dim=0) # 2 X T X 24 X 3
    image_directory = 'outputs/test/test_smpl_to_amass'
    # Visualize single seq           
    print(f'start visualize...')
    show3Dpose_animation_multiple(concat_seq_cmp.data.cpu().numpy(), image_directory, 0, "cmp_vibe_ours_dance_vis", str(3), use_amass=True)


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

