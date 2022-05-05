'''
Description: 
Author: JinShiyin
Email: shiyinjin@foxmail.com
Date: 2022-04-19 00:31:00
'''
import os
import time
import numpy as np
from lib.datasets.amass import AMASS
from lib.models.fk_layer import ForwardKinematicsLayer

from lib.utils.common_utils import init_config

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
# from lib.utils.demo_utils import images_to_video
# import os

# img_root = '/data/jsy/datasets/3DPW/imageFiles'
# img_folder_name_list = os.listdir(img_root)
# out_dir = '/data/jsy/datasets/3DPW/videos'
# image_name_pattern = f'image_%05d.jpg'
# for img_folder_name in img_folder_name_list:
#     img_folder = os.path.join(img_root, img_folder_name)
#     out_video_path = os.path.join(out_dir, f'{img_folder_name}.mp4')
#     images_to_video(img_folder, out_video_path, image_name_pattern)

# def process1(logger):
#     print(f'process1')
#     logger.info(f'process1')

# def process2(logger):
#     print(f'process2')
#     logger.info(f'process2')


# if __name__ == '__main__':
#     # from torch.multiprocessing import Process, set_start_method
#     # set_start_method('spawn')

#     from multiprocessing import Process

#     from lib.utils.logs import init_logger

#     logger = init_logger(log_name='test', log_dir='/data/jsy/code/hm-vae/outputs/test/tmp')

#     logger.info(f'main process')

#     p = Process(target=process1, args=(logger,))
#     p.start()

#     p = Process(target=process2, args=(logger,))
#     p.start()

#####################################
import torch

from lib.utils.conversion_utils import convert_to_input, get_all_contact_label
from utils_common import show3Dpose_animation_with_contact_label

fk_layer = ForwardKinematicsLayer()
cfg_path = 'configs/MAE/config.yaml'
cfg = init_config(cfg_path)


test_dataset = AMASS(seqlen=128, overlap=cfg['overlap'], db_dir=cfg['db_dir'], mode='test', logger=None)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True,
    num_workers=1, pin_memory=True, drop_last=True)

for test_it, data in enumerate(test_loader):
    if test_it>10:
        break
    data = data.cuda()
    print(f'test_it={test_it}, data.shape={data.shape}')

    seq_rot_6d, seq_rot_mat, seq_rot_pos, seq_trans, seq_linear_v, seq_root_v = convert_to_input(
        data=data,
        fk_layer=fk_layer,
        mean_std_data=None
    )
    contact_label = get_all_contact_label(
        seq_rot_pos, 
        seq_trans, 
        pos_height_thresh=0.045, 
        velocity_thresh=0.1
    )
    print(contact_label.size())
    # exit()
    
    bs, timesteps, _ = seq_rot_pos.size()
    seq_rot_pos = seq_rot_pos.view(bs, timesteps, 24, 3) # bs X T X 24 X 3
    seq_trans = seq_trans.unsqueeze(2) # bs X T X 1 X 3
    absolute_rot_pos = seq_rot_pos + seq_trans
    absolute_rot_pos = absolute_rot_pos.cpu().numpy() # bs X T X 1 X 3
    contact_label = contact_label.cpu().numpy()
    dst_path = f'outputs/test/contact_ground_vis_mean/{test_it:02d}.gif'
    show3Dpose_animation_with_contact_label(
        channels=absolute_rot_pos,
        contact_label=contact_label,
        use_joint12=False,
        use_amass=True,
        use_lafan=False,
        dest_vis_path=dst_path
    )
    print(f'{dst_path} saved...')


###################

# import matplotlib
# matplotlib.use('Agg')
# from matplotlib import pyplot as plt
# import numpy as np
# from matplotlib import animation

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')

# # create the parametric curve
# frame_num = 100
# x = np.random.random(frame_num)
# y = np.random.random(frame_num)
# z = np.random.random(frame_num)

# # create the first plot
# point, = ax.plot([], [], [], 'r.')
# point2, =ax.plot([], [], [], '.')
# # line, = ax.plot(x, y, z, label='parametric curve')
# # ax.legend()
# ax.set_xlim([0, 1])
# ax.set_ylim([0, 1])
# ax.set_zlim([0, 1])

# # second option - move the point position at every frame
# def update_point(n, x, y, z, point):
#     point2.set_data([x[0:n], y[0:n]])
#     point2.set_3d_properties(z[0:n], 'z')
#     point.set_data([x[n], y[n]])
#     point.set_3d_properties(z[n], 'z')
#     # point.set_array([255,0,0])
#     return point

# ani=animation.FuncAnimation(fig, update_point, frame_num, fargs=(x, y, z, point))
# ani.save('outputs/test/tmp/test1.gif', writer='pillow', fps=1)



