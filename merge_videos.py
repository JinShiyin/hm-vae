'''
Description: 
Author: JinShiyin
Email: shiyinjin@foxmail.com
Date: 2022-04-25 21:22:14
'''
import os
from lib.utils.merge_video_utils import merge_videos


def test():
    video_name_list = [
        'downtown_walkUphill_00.mp4', 
        'outdoors_fencing_01.mp4', 
        'outdoors_freestyle_01.mp4', 
        'downtown_weeklyMarket_00.mp4'
    ]
    ori_video_dir = 'outputs/input_videos'
    merged_video_dir = 'outputs/render/merged_videos/videos/without_ori_video'
    os.makedirs(merged_video_dir, exist_ok=True)

    refined_video_dir = 'outputs/render/rendered_videos/refined/without_ori_video'
    vibe_video_dir = 'outputs/render/rendered_videos/vibe/without_ori_video'

    for video_name in video_name_list:
        video_basename = os.path.splitext(video_name)[0]
        video_path_list = []
        video_path_list.append(os.path.join(ori_video_dir, video_name))
        video_path_list.append(os.path.join(vibe_video_dir, f'{video_basename}_rendered.mp4'))
        video_path_list.append(os.path.join(refined_video_dir, f'{video_basename}_rendered.mp4'))
        tag_list = ['Input', 'VIBE', 'Ours']
        output_video_path = os.path.join(merged_video_dir, f'{video_basename}_merged.mp4')
        merge_videos(
            video_path_list=video_path_list,
            tag_list=tag_list,
            output_video_path=output_video_path
        )


if __name__ == '__main__':
    test()

