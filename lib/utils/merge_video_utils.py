'''
Description: 
Author: JinShiyin
Email: shiyinjin@foxmail.com
Date: 2022-04-25 19:39:17
'''
import os
import cv2
import sys
import shutil
from tqdm import tqdm
from lib.utils.demo_utils import video_to_images, images_to_video

def merge_videos(video_path_list, tag_list, output_video_path, tmp_folder='outputs/render/merged_videos/tmp_for_images'):
    video_info_list = []
    min_height = sys.maxsize
    min_frame_num = sys.maxsize
    for i in range(len(video_path_list)):
        video_path = video_path_list[i]
        img_folder, frame_num, img_shape = video_to_images(video_path, data_folder=tmp_folder, return_info=True, flag=f'{i:02d}')
        height, width, _ = img_shape
        min_height = min(min_height, height)
        min_frame_num = min(min_frame_num, frame_num)
        video_info_list.append({
            'img_folder': img_folder,
            'frame_num': frame_num,
            'img_shape': [width, height]
        })
    
    for i in range(len(video_info_list)):
        width, height = video_info_list[i]['img_shape']
        video_info_list[i]['img_shape'] = [int(width*(min_height/height)), min_height]
    
    merged_img_dir = os.path.join(tmp_folder, 'merged_images')
    os.makedirs(merged_img_dir, exist_ok=True)
    
    for i in tqdm(range(1, min_frame_num), ncols=100):
        img_list = []
        for j in range(len(video_info_list)):
            video_info = video_info_list[j]
            img_path = os.path.join(video_info['img_folder'], f'{i:06d}.png')
            img = cv2.imread(img_path)
            img = cv2.resize(img, dsize=video_info['img_shape'])
            cv2.putText(img, tag_list[j], (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
            img_list.append(img)
        img = cv2.hconcat(img_list)
        output_img_path = os.path.join(merged_img_dir, f'{i:06d}.jpg')
        cv2.imwrite(output_img_path, img)
    images_to_video(merged_img_dir, output_video_path, image_name_pattern=f'%06d.jpg')

    print(f'clear the tmp dir')
    for video_info in video_info_list:
        img_folder = video_info['img_folder']
        shutil.rmtree(img_folder)
        print(f'delete {img_folder}')
    shutil.rmtree(merged_img_dir)
    print(f'delete {merged_img_dir}')
