'''
Description: 
Author: JinShiyin
Email: shiyinjin@foxmail.com
Date: 2022-04-22 15:29:11
'''
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

def render_result_with_origin_video(ori_video_path, output_video_folder, cam, rot_mat):
    '''
    description: 
    param {*} ori_video_path
    param {*} output_video_folder
    param {torch.floatTensor} cam: T X 4
    param {torch.floatTensor} rot_mat: T X 24 X 3 X 3
    return {*}
    '''

    data_folder = 'outputs/render/video_imgs_folder'
    os.makedirs(data_folder, exist_ok=True)
    image_folder, frame_nums, img_shape = video_to_images(ori_video_path, data_folder, return_info=True)

    # set renderer
    renderer = Renderer(resolution=(img_shape[1], img_shape[0]), orig_img=True)

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
        cam = torch.tensor([0.5, 0.8, 0, 0]).float().expand(timesteps, 4).cuda()
    
    verts = verts.cpu().numpy()
    cam = cam.cpu().numpy()

    output_img_folder = f'{image_folder}_output'
    os.makedirs(output_img_folder, exist_ok=True)
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
        cv2.imwrite(os.path.join(output_img_folder, f'{frame_idx:06d}.png'), img)
    
    os.makedirs(output_video_folder, exist_ok=True)
    save_name = f'{os.path.splitext(os.path.basename(ori_video_path))[0]}_rendered.mp4'
    output_video_path = os.path.join(output_video_folder, save_name)
    print(f'Saving result video to {output_video_path}')
    images_to_video(img_folder=output_img_folder, output_vid_file=output_video_path)
    print('================= END =================')


def test1():
    video_name = 'hiphop_clip1' # sample_video, hiphop_clip1
    ori_video_path = f'/data/jsy/code/VIBE/{video_name}.mp4'
    output_video_folder = 'outputs/render/video_folder'
    vibe_path = f'/data/jsy/code/VIBE/output/{video_name}/vibe_output.pkl'
    vibe_data = joblib.load(vibe_path)
    cam = torch.from_numpy(vibe_data[1]['orig_cam']).float().cuda() # T X 4
    # npy_path = 'outputs/test/refine_vibe/20220423-215309-sample_video/1_our_rot_mat.npy'
    npy_path = 'outputs/test/refine_vibe/20220423-220912-hiphop_clip1/1_our_rot_mat.npy'
    rot_mat = np.load(npy_path) # T X 24 X 3 X 3
    rot_mat = torch.from_numpy(rot_mat).float().cuda()
    render_result_with_origin_video(ori_video_path, output_video_folder, cam, rot_mat)


if __name__ == '__main__':
    test1()
    

        

        









