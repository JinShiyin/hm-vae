'''
Description: 
Author: JinShiyin
Email: shiyinjin@foxmail.com
Date: 2022-04-19 00:31:00
'''
import os
import time
import numpy as np

# time_stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
# print(time_stamp)

# joints_to_use = np.array([
#     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
#     11, 12, 13, 14, 15, 16, 17, 18, 19,
#     20, 21, 22, 37
# ])
# joints_to_use = np.arange(0, 156).reshape((-1, 3))[joints_to_use].reshape(-1)
# print(joints_to_use)

# npz_path = f"/data/jsy/datasets/AMASS/amass_for_hm_vae_fps30/CMU_55_55_24_poses.npy"
# data = np.load(npz_path, allow_pickle=True)
# print(data)
# print(data.shape)
# # for key, val in data.items():
# #     print(key)
# #     print(val)
# #     print(val.shape)

# amass_dir = '/data/jsy/datasets/AMASS/decompress'
# sequences = os.listdir(amass_dir)
# print(sequences)

# import torch
# print(torch.Size([1, 1]))

# ###################################### test smpl model ########################
# import torch
# import time
# from lib.models.smpl import SMPL, VIBE_DATA_DIR

# batch_size = 301
# smpl = SMPL(
#     VIBE_DATA_DIR,
#     batch_size=batch_size,
#     create_transl=False
# )
# smpl.cuda()
# smpl.eval()
# betas = torch.zeros(batch_size, 10).float().cuda()
# body_pose = torch.zeros(batch_size, 23, 3, 3).float().cuda()
# global_orient = torch.zeros(batch_size, 1, 3, 3).float().cuda()

# while True:
#     with torch.no_grad():
#         start_time = time.time()
#         smpl_output = smpl(
#             betas=betas,
#             body_pose=body_pose, # T X 23 X 3 X 3
#             global_orient=global_orient, # T X 1 X 3 X 3
#             pose2rot=False
#         )
#         verts = smpl_output.vertices
#         print(verts.shape)
#         print(f'used_time = {time.time()-start_time}')


# ############################## generate 3dpw videos ###################
from lib.utils.demo_utils import images_to_video
import os

img_root = '/data/jsy/datasets/3DPW/imageFiles'
img_folder_name_list = os.listdir(img_root)
out_dir = '/data/jsy/datasets/3DPW/videos'
image_name_pattern = f'image_%05d.jpg'
for img_folder_name in img_folder_name_list:
    img_folder = os.path.join(img_root, img_folder_name)
    out_video_path = os.path.join(out_dir, f'{img_folder_name}.mp4')
    images_to_video(img_folder, out_video_path, image_name_pattern)

