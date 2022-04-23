'''
Description: 
Author: JinShiyin
Email: shiyinjin@foxmail.com
Date: 2022-04-19 00:31:00
'''
import os
import time
import numpy as np

time_stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
print(time_stamp)

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

import torch
print(torch.Size([1, 1]))


