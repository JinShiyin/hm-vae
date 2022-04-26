# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import cv2
import subprocess

def video_to_images(vid_file, data_folder=None, return_info=False, flag=None):
    img_folder = os.path.join(data_folder, os.path.basename(vid_file).replace('.', '_')) + flag
    print(f'{vid_file} will save frames to {img_folder}')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder, exist_ok=True)
        command = ['ffmpeg',
                '-i', vid_file,
                '-f', 'image2',
                '-v', 'error',
                f'{img_folder}/%06d.png']
        print(f'Running \"{" ".join(command)}\"')
        subprocess.call(command)
        print(f'Images saved to \"{img_folder}\"')
    else:
        print(f'{img_folder} already exists')

    img_shape = cv2.imread(os.path.join(img_folder, '000001.png')).shape

    if return_info:
        return img_folder, len(os.listdir(img_folder)), img_shape
    else:
        return img_folder


def images_to_video(img_folder, output_vid_file, image_name_pattern=None):
    os.makedirs(img_folder, exist_ok=True)

    if image_name_pattern is None:
        image_name_pattern = f'%06d.png'
    command = [
        'ffmpeg', '-y', '-threads', '16', '-i', f'{img_folder}/'+image_name_pattern, '-profile:v', 'baseline',
        '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-v', 'error', output_vid_file,
    ]

    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)
