'''
Description: 
Author: JinShiyin
Email: shiyinjin@foxmail.com
Date: 2022-04-21 11:21:24
'''
import os
import torch
import time
from utils_common import get_config, show3Dpose_animation
from seq_two_hier_sa_vae import TwoHierSAVAEModel
from utils_motion_vae import get_train_loaders_all_data_seq

cfg_path = 'configs/len64_no_aug_hm_vae.yaml'
config = get_config(cfg_path)

# print(f'build model...')
model = TwoHierSAVAEModel(config)
# model_path = 'outputs/len64_no_aug_hm_vae/20220426-173227/checkpoints/gen_00040000.pt'
# model_path = 'outputs/len64_no_aug_hm_vae/20220422-114625/checkpoints/gen_01250000.pt'
model_path = 'outputs/len64_no_aug_hm_vae/20220427-112637/checkpoints/gen_00060000.pt' # seq_len=8
print(f'load checkpoint from {model_path}...')
model.load_state_dict(torch.load(model_path)['state_dict'])
model.cuda()
model.eval()

# refine vibe
video_name = 'downtown_weeklyMarket_00' # hiphop_clip1, sample_video, downtown_walkUphill_00, outdoors_fencing_01, outdoors_freestyle_01, downtown_weeklyMarket_00
vibe_path = f'/data/jsy/code/VIBE/output/{video_name}/vibe_output.pkl'
time_stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
# time_stamp = "0-test"
output_root = 'outputs/test/refine_vibe'
output_dir = os.path.join(output_root, f'{time_stamp}-{video_name}')
os.makedirs(output_dir, exist_ok=True)
print('start refine vibe...')
model.refine_dance_motions(vibe_path=vibe_path, image_directory=output_dir)
# model.refine_dance_motions_not_center_strategy(vibe_path=vibe_path, image_directory=output_dir)


# # refine test dataset
# config['batch_size'] = 1
# config['train_seq_len'] = 301
# data_loaders = get_train_loaders_all_data_seq(config)
# test_loader = data_loaders[2]
# time_stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
# output_dir = os.path.join('outputs/test/refine_amass', time_stamp)
# os.makedirs(output_dir, exist_ok=True)
# print('start refine amass...')
# for test_it, test_input_data in enumerate(test_loader):
#     if test_it >=1:
#         break
#     seq_rot_6d, seq_rot_mat, seq_rot_pos, seq_joint_pos, seq_linear_v, seq_angular_v, seq_root_v = test_input_data
#     print(test_it)
#     print(seq_rot_6d.size())
#     model.refine_amass_motions(seq_rot_6d, image_directory=output_dir)


# visualize
# print(f'start visulization...')
# config['batch_size'] = 1
# config['train_seq_len'] = 64
# iterations = 0
# time_stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
# image_directory = os.path.join('outputs/test/rec_amass', time_stamp)
# os.makedirs(image_directory, exist_ok=True)

# data_loaders = get_train_loaders_all_data_seq(config)
# test_loader = data_loaders[2]
# with torch.no_grad():
#     for test_it, test_input_data in enumerate(test_loader):

#         # Generate long sequences
#         gt_seq, mean_seq_out_pos, sampled_seq_out_pos, \
#             zero_seq_out_pos \
#             = model.gen_seq(test_input_data, config, iterations=0)
#         # T X bs X 24 X 3, T X bs X 24 X 3, T X bs X 24 X 3, T X bs X 24 X 3
#         for bs_idx in range(0, 1):  # test data loader set bs to 1
#             # 1 X T X 24 X 3
#             gt_seq_for_vis = gt_seq[:, bs_idx, :, :][None, :, :, :]

#             # T X 24 X 3
#             mean_out_for_vis = mean_seq_out_pos[:, bs_idx, :, :]
#             # 1 X T X 24 X 3
#             mean_out_for_vis = mean_out_for_vis[None, :, :, :]

#             cat_gt_mean_rot_seq_for_vis = torch.cat(
#                 (gt_seq_for_vis, mean_out_for_vis), dim=0)

#             # T X 24 X 3
#             sampled_out_for_vis = sampled_seq_out_pos[:,
#                                                         bs_idx, :, :]
#             # 1 X T X 24 X 3
#             sampled_out_for_vis = sampled_out_for_vis[None, :, :, :]

#             if config['model_name'] == "TrajectoryModel":
#                 show3Dpose_animation(cat_gt_mean_rot_seq_for_vis.data.cpu().numpy(), image_directory,
#                                         iterations, "mean_seq_rot_6d", test_it, use_amass=True)
#                 show3Dpose_animation(sampled_out_for_vis.data.cpu().numpy(), image_directory,
#                                         iterations, "sampled_seq_rot_6d", test_it, use_amass=True)
#             else:
#                 if config['random_root_rot_flag']:
#                     show3Dpose_animation(cat_gt_mean_rot_seq_for_vis.data.cpu().numpy(), image_directory,
#                                             iterations, "mean_seq_rot_6d", test_it, use_amass=False)
#                     show3Dpose_animation(sampled_out_for_vis.data.cpu().numpy(), image_directory,
#                                             iterations, "sampled_seq_rot_6d", test_it, use_amass=False)
#                 else:
#                     show3Dpose_animation(cat_gt_mean_rot_seq_for_vis.data.cpu().numpy(), image_directory,
#                                             iterations, "mean_seq_rot_6d", test_it, use_amass=True)
#                     show3Dpose_animation(sampled_out_for_vis.data.cpu().numpy(), image_directory,
#                                             iterations, "sampled_seq_rot_6d", test_it, use_amass=True)

