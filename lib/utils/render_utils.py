'''
Description: 
Author: JinShiyin
Email: shiyinjin@foxmail.com
Date: 2022-04-28 14:32:00
'''
import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import cv2
import time
import torch
import shutil
import colorsys
import numpy as np
from tqdm import tqdm
from lib.models.smpl import SMPL, VIBE_DATA_DIR
from lib.utils.renderer import Renderer
from lib.utils.demo_utils import images_to_video, video_to_images
from utils_common import show3Dpose_animation_multiple

SMPL_MODEL_DIR = VIBE_DATA_DIR


def render_result_without_origin_video(rendered_image_folder, output_video_path, cam, rot_mat, resolution, logger):
    '''
    description: 
    param {*} rendered_image_folder
    param {*} output_video_path
    param {torch.floatTensor} cam: T X 4
    param {torch.floatTensor} rot_mat: T X 24 X 3 X 3
    param {*} resolution: (width, height) 
    return {*}
    '''

    # set renderer
    renderer = Renderer(resolution=resolution, orig_img=False)

    # set saml model and calculate the verts given rot_mat and default shape
    timesteps, n_joints, _, _ = rot_mat.size()
    betas = torch.zeros(timesteps, 10).float().cuda()

    print(f'PID={os.getpid()}, start cal verts using smpl model...')
    smpl = SMPL(
        SMPL_MODEL_DIR,
        batch_size=timesteps,
        create_transl=False
    )
    smpl.cuda()
    smpl.eval()
    with torch.no_grad():
        smpl_output = smpl(
            betas=betas,
            body_pose=rot_mat[:, 1:], # T X 23 X 3 X 3
            global_orient=rot_mat[:, [0]], # T X 1 X 3 X 3
            pose2rot=False
        )
    verts = smpl_output.vertices # T X 6890 X 3

    # set default cam
    if cam is None:
        # cam = torch.tensor([0.46, 0.82, 0.01, 0.36]).float().expand(timesteps, 4).cuda()
        # sx / sy = resolution[1] / resolution[0], i.e. sx * width = sy * height, 即 sx 与 sy 控制了视锥体的长宽
        cam = torch.tensor([0.7*resolution[1]/resolution[0], 0.7, 0, 0]).float().expand(timesteps, 4).cuda()
    
    verts = verts.cpu().numpy()
    cam = cam.cpu().numpy()
    mesh_color = colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)

    print(f'PID={os.getpid()}, start render...')
    for frame_idx in tqdm(range(timesteps), ncols=100):
        img = renderer.render(
            None,
            verts[frame_idx], # # 6890 X 3
            cam=cam[frame_idx],
            color=mesh_color,
            mesh_filename=None,
        )
        cv2.imwrite(os.path.join(rendered_image_folder, f'{frame_idx:06d}.png'), img)
    
    print(f'PID={os.getpid()}, Saving result video to {output_video_path}')
    images_to_video(img_folder=rendered_image_folder, output_vid_file=output_video_path)
    print(f'PID={os.getpid()}, ================= END =================')


def render_multi_refined_rot_mat(cfg, refined_rot_mat_dir, output_dir, logger):
    start_time = time.time()
    print(f'Render process start, PID={os.getpid()}, refined_rot_mat_dir={refined_rot_mat_dir}, output_dir={output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    print(f'PID={os.getpid()}, makedir {output_dir}')
    video_name_list = cfg['video_list']
    tmp_rendered_image_folder = cfg['tmp_rendered_image_folder']
    for video_name in video_name_list:
        print(f'PID={os.getpid()}, start render {video_name}')
        video_basename = os.path.splitext(video_name)[0]
        rendered_image_folder = os.path.join(tmp_rendered_image_folder, video_basename)
        os.makedirs(rendered_image_folder, exist_ok=True)
        refined_rot_mat_path = os.path.join(refined_rot_mat_dir, f'refined_{video_basename}_rot_mat.npy')
        output_video_path = os.path.join(output_dir, f'{video_basename}.mp4')
        rot_mat = torch.from_numpy(np.load(refined_rot_mat_path)).float().cuda()
        render_result_without_origin_video(
            rendered_image_folder=rendered_image_folder,
            output_video_path=output_video_path,
            cam=None,
            rot_mat=rot_mat,
            resolution=cfg['resolution'],
            logger=logger
        )
        torch.cuda.empty_cache()
        shutil.rmtree(rendered_image_folder)
        print(f'PID={os.getpid()}, Finish render {video_name}, delete {rendered_image_folder}')
    print(f'Render process finish, PID={os.getpid()}, used_time={time.time()-start_time}')
    

def show3d_multi_refined_rot_pos(cfg, refined_rot_pos_dir, image_directory, iterations, tag, logger):
    start_time = time.time()
    print(f'Show3d process start, PID={os.getpid()}, refined_rot_pos_dir={refined_rot_pos_dir}, image_directory={image_directory}, iterations={(iterations+1)}, tag={tag}')
    video_name_list = os.listdir(refined_rot_pos_dir)
    video_name_list.sort()
    for video_name in video_name_list:
        print(f'PID={os.getpid()}, start show 3d {video_name}')
        video_basename = os.path.splitext(video_name)[0]
        concat_seq_cmp_path = os.path.join(refined_rot_pos_dir, f'{video_basename}.npy')
        concat_seq_cmp = np.load(concat_seq_cmp_path)
        show3Dpose_animation_multiple(
            channels=concat_seq_cmp,
            image_directory=image_directory, 
            iterations=(iterations+1), 
            tag=tag, 
            bs_idx=video_basename, 
            use_joint12=False, 
            use_amass=True, 
            use_lafan=False, 
            make_translation=False
        )
        gif_path = os.path.join(image_directory, str((iterations+1)), tag, f'{video_basename}.gif')
        print(f'PID={os.getpid()}, {gif_path} saved')
    print(f'Show3d Process finish, PID={os.getpid()}, used_time={time.time()-start_time}')
