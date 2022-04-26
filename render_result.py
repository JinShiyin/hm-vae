'''
Description: 
Author: JinShiyin
Email: shiyinjin@foxmail.com
Date: 2022-04-22 15:29:11
'''
import argparse
import os

os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import cv2
import torch
import joblib
import colorsys
import numpy as np
from tqdm import tqdm
from lib.models.smpl import SMPL, VIBE_DATA_DIR
from lib.utils.renderer import Renderer
from lib.utils.demo_utils import images_to_video, video_to_images

SMPL_MODEL_DIR = VIBE_DATA_DIR

def render_result_with_origin_video(ori_video_path, rendered_image_folder, output_video_path, cam, rot_mat):
    '''
    description: 
    param {*} ori_video_path
    param {*} rendered_image_folder
    param {*} output_video_path
    param {torch.floatTensor} cam: T X 4
    param {torch.floatTensor} rot_mat: T X 24 X 3 X 3
    return {*}
    '''

    data_folder = 'outputs/render/ori_images'
    os.makedirs(data_folder, exist_ok=True)
    image_folder, frame_nums, img_shape = video_to_images(ori_video_path, data_folder, return_info=True)
    height, width = img_shape[0], img_shape[1]

    # set renderer
    renderer = Renderer(resolution=(width, height), orig_img=True)

    # set saml model and calculate the verts given rot_mat and default shape
    timesteps, n_joints, _, _ = rot_mat.size()
    betas = torch.zeros(timesteps, 10).float().cuda()

    print(f'start cal verts using smpl model...')
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
        cam = torch.tensor([0.7*height/width, 0.7, 0, 0]).float().expand(timesteps, 4).cuda()
    
    verts = verts.cpu().numpy()
    cam = cam.cpu().numpy()

    mesh_color = colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)
    image_file_names = sorted([
        os.path.join(image_folder, x)
        for x in os.listdir(image_folder)
        if x.endswith('.png') or x.endswith('.jpg')
    ])

    print(f'start render...')
    for frame_idx in tqdm(range(frame_nums), ncols=100):
        img_fname = image_file_names[frame_idx]
        img = cv2.imread(img_fname)

        img = renderer.render(
            img,
            verts[frame_idx], # # 6890 X 3
            cam=cam[frame_idx],
            color=mesh_color,
            mesh_filename=None,
        )
        cv2.imwrite(os.path.join(rendered_image_folder, f'{frame_idx:06d}.png'), img)
    
    print(f'Saving result video to {output_video_path}')
    images_to_video(img_folder=rendered_image_folder, output_vid_file=output_video_path)
    print('================= END =================')



def render_result_without_origin_video(rendered_image_folder, output_video_path, cam, rot_mat, resolution):
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

    print(f'start cal verts using smpl model...')
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

    print(f'start render...')
    for frame_idx in tqdm(range(timesteps), ncols=100):
        img = renderer.render(
            None,
            verts[frame_idx], # # 6890 X 3
            cam=cam[frame_idx],
            color=mesh_color,
            mesh_filename=None,
        )
        cv2.imwrite(os.path.join(rendered_image_folder, f'{frame_idx:06d}.png'), img)
    
    print(f'Saving result video to {output_video_path}')
    images_to_video(img_folder=rendered_image_folder, output_vid_file=output_video_path)
    print('================= END =================')


def test1():
    video_name = 'downtown_walkUphill_00' # sample_video, hiphop_clip1, downtown_walkUphill_00
    ori_video_path = f'/data/jsy/code/VIBE/{video_name}.mp4'
    output_video_folder = 'outputs/render/video_folder'
    vibe_path = f'/data/jsy/code/VIBE/output/{video_name}/vibe_output.pkl'
    vibe_data = joblib.load(vibe_path)
    cam = torch.from_numpy(vibe_data[1]['orig_cam']).float().cuda() # T X 4
    npy_path = 'outputs/test/refine_vibe/20220425-152931-downtown_walkUphill_00/1_our_rot_mat.npy'
    # npy_path = 'outputs/test/refine_vibe/20220423-220912-hiphop_clip1/1_our_rot_mat.npy'
    rot_mat = np.load(npy_path) # T X 24 X 3 X 3
    rot_mat = torch.from_numpy(rot_mat).float().cuda()
    # render_result_with_origin_video(ori_video_path, os.path.join(output_video_folder, 'with_ori_video'), cam, rot_mat)

    render_result_without_origin_video(
        output_image_folder=f'outputs/render/video_imgs_folder/{video_name}_mp4_output',
        output_video_path=f'outputs/render/video_folder/without_ori_video/{video_name}_rendered.mp4',
        cam=None,
        rot_mat=rot_mat,
        resolution=(1280, 1280)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refined_data_path', type=str, help='the refined npy path of HM-VAE')
    parser.add_argument('--vibe_data_path', default=None, type=str, help='the output pkl path of vibe')
    parser.add_argument('--ori_video_path', type=str, default=None, help='the original video path')
    parser.add_argument('--use_vibe_cam', action='store_true', help='if use the cam param')
    parser.add_argument('--use_ori_img', action='store_true', help='if use original image')
    parser.add_argument('--resolution', type=int, default=1280, help='the resolution of rendered image')
    parser.add_argument('--rendered_image_folder', type=str, help='the output dir of rendered image')
    parser.add_argument('--output_video_path', type=str, help='the rendered video path')
    args = parser.parse_args()

    vibe_data = joblib.load(args.vibe_data_path) if os.path.exists(args.vibe_data_path) else None
    cam = torch.from_numpy(vibe_data[1]['orig_cam']).float().cuda() if args.use_vibe_cam else None
    rot_mat = torch.from_numpy(np.load(args.refined_data_path)).float().cuda() # T X 24 X 3 X 3
    os.makedirs(args.rendered_image_folder, exist_ok=True)
    print(f'create rendered image folder: {args.rendered_image_folder}')

    if args.use_ori_img:
        print(f'use_ori_img={args.use_ori_img}, use_vibe_cam={args.use_vibe_cam}, start render with original image...')
        render_result_with_origin_video(
            ori_video_path=args.ori_video_path,
            rendered_image_folder= args.rendered_image_folder,
            output_video_path=args.output_video_path,
            cam=cam,
            rot_mat=rot_mat
        )
    else:
        print(f'use_ori_img={args.use_ori_img}, use_vibe_cam={args.use_vibe_cam}, start render without original image...')
        render_result_without_origin_video(
            rendered_image_folder=args.rendered_image_folder,
            output_video_path=args.output_video_path, 
            cam=cam, 
            rot_mat=rot_mat, 
            resolution=(args.resolution, args.resolution)
        )


if __name__ == '__main__':
    # test1()
    main()
