'''
Description: 
Author: JinShiyin
Email: shiyinjin@foxmail.com
Date: 2022-04-25 16:08:17
'''
import os
import subprocess

def main():
    vibe_root = '/data/jsy/code/VIBE/output'
    ori_video_dir = 'outputs/input_videos'
    output_root = 'outputs/render'
    # video_name_list = [
    #     'sample_video.mp4', 
    #     'downtown_walkUphill_00.mp4', 
    #     'outdoors_fencing_01.mp4', 
    #     'outdoors_freestyle_01.mp4', 
    #     'hiphop_clip1.mp4', 
    #     'downtown_weeklyMarket_00.mp4'
    # ]
    refine_data_info_list = [
        {
            'refined_data_path': 'outputs/test/refine_vibe/20220427-210933-sample_video/1_our_rot_mat.npy',
            'vibe_data_path': 'outputs/test/refine_vibe/20220427-210933-sample_video/1_vibe_rot_mat.npy',
            'video_name': 'sample_video.mp4',
        },
        {
            'refined_data_path': 'outputs/test/refine_vibe/20220427-210022-hiphop_clip1/1_our_rot_mat.npy',
            'vibe_data_path': 'outputs/test/refine_vibe/20220427-210022-hiphop_clip1/1_vibe_rot_mat.npy',
            'video_name': 'hiphop_clip1.mp4',
        },
        {
            'refined_data_path': 'outputs/test/refine_vibe/20220427-210612-downtown_walkUphill_00/1_our_rot_mat.npy',
            'vibe_data_path': 'outputs/test/refine_vibe/20220427-210612-downtown_walkUphill_00/1_vibe_rot_mat.npy',
            'video_name': 'downtown_walkUphill_00.mp4',
        },
        {
            'refined_data_path': 'outputs/test/refine_vibe/20220427-205614-outdoors_fencing_01/1_our_rot_mat.npy', # seq-len=16
            'vibe_data_path': 'outputs/test/refine_vibe/20220427-205614-outdoors_fencing_01/1_our_rot_mat.npy',
            'video_name': 'outdoors_fencing_01.mp4',
        },
        {
            'refined_data_path': 'outputs/test/refine_vibe/20220427-210212-outdoors_freestyle_01/1_our_rot_mat.npy',
            'vibe_data_path': 'outputs/test/refine_vibe/20220427-210212-outdoors_freestyle_01/1_vibe_rot_mat.npy',
            'video_name': 'outdoors_freestyle_01.mp4',
        },
        {
            'refined_data_path': 'outputs/test/refine_vibe/20220427-211216-downtown_weeklyMarket_00/1_our_rot_mat.npy',
            'vibe_data_path': 'outputs/test/refine_vibe/20220427-211216-downtown_weeklyMarket_00/1_vibe_rot_mat.npy',
            'video_name': 'downtown_weeklyMarket_00.mp4',
        },
    ]

    use_vibe_cam = False
    use_ori_img = False
    resolution = 1280
    is_render_refined = True # True->refined, False->vibe
    render_flag = 'refined' if is_render_refined else 'vibe'
    for refine_data_info in refine_data_info_list:
        if is_render_refined:
            refined_data_path = refine_data_info['refined_data_path']
        else:
            refined_data_path = refine_data_info['vibe_data_path']
        video_name = refine_data_info['video_name']
        video_basename = os.path.splitext(video_name)[0]
        ori_video_path = os.path.join(ori_video_dir, video_name)
        vibe_data_path = os.path.join(vibe_root, video_basename, 'vibe_output.pkl')
        

        rendered_image_folder = os.path.join(output_root, 'rendered_images', render_flag, video_name.replace('.', '_')+'_output')
        os.makedirs(rendered_image_folder, exist_ok=True)
        render_videos_folder_name = 'with_ori_video' if use_ori_img else 'without_ori_video'
        output_video_folder = os.path.join(output_root, 'rendered_videos', render_flag, render_videos_folder_name)
        os.makedirs(output_video_folder, exist_ok=True)
        output_video_path = os.path.join(output_video_folder, video_basename+'_rendered.mp4')
        command = f'CUDA_VISIBLE_DEVICES=0 python render_result.py --refined_data_path {refined_data_path} --vibe_data_path {vibe_data_path} --ori_video_path {ori_video_path} --resolution {resolution} --rendered_image_folder {rendered_image_folder} --output_video_path {output_video_path}'
        if use_vibe_cam:
            command += '--use_vibe_cam'
        if use_ori_img:
            command += '--use_ori_img'
        
        print(f'Running {command}')
        os.system(command)

if __name__ == '__main__':
    main()


