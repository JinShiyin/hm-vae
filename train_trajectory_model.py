'''
Description: 
Author: JinShiyin
Email: shiyinjin@foxmail.com
Date: 2022-05-01 14:57:21
'''
'''
Description: 
Author: JinShiyin
Email: shiyinjin@foxmail.com
Date: 2022-04-26 21:56:17
'''
import os
from lib.models.fk_layer import ForwardKinematicsLayer

from lib.utils.conversion_utils import convert_to_input
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import sys
import time
import torch
import shutil
import argparse
from torch.multiprocessing import set_start_method
from multiprocessing import Process
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import numpy as np

from lib.utils.logs import init_logger
from lib.utils.common_utils import setup_seed, write_loss
from lib.datasets.amass import get_all_loaders
from lib.utils.common_utils import init_config, create_sub_folders
from lib.utils.render_utils import render_multi_refined_rot_mat, show3d_multi_refined_rot_pos
from lib.trainer.trajectory_trainer import TrajectoryModelTrainer


if __name__ == '__main__':
    # Enable auto-tuner to find the best algorithm to use for your hardware.
    cudnn.benchmark = True

    set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/TrajectoryModel/config.yaml', help='configuration file for training and testing')
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Load experiment setting
    config = init_config(args.config)
    max_iter = config['max_iter']

    # Setup logger and output folders
    output_dir = config['output_dir']
    logger = init_logger(log_name='TrajectoryModel', log_dir=output_dir)
    log_dir, checkpoint_directory, image_directory = create_sub_folders(output_dir)
    if not args.resume:
        shutil.copyfile(args.config, os.path.join(output_dir, 'config.yaml'))
    train_writer = SummaryWriter(log_dir)

    trainer = TrajectoryModelTrainer(config, logger)
    trainer.cuda()
    iterations = trainer.resume(checkpoint_directory) if args.resume else 0

    logger.info(f'Start load amass data')
    data_loaders = get_all_loaders(config, logger)
    logger.info(f'Finish load amass data')
    train_loader = data_loaders[0]
    val_loader = data_loaders[1]
    test_loader = data_loaders[2]

    epoch = 0
    while True:
        epoch+=1
        train_dataset = train_loader
        val_dataset = val_loader
        test_dataset = test_loader
        len_train = len(train_dataset)
        len_val = len(val_dataset)
        len_test = len(test_dataset)

        #  train
        for it, input_data in enumerate(train_dataset):
            input_data = input_data.cuda() # bs X T X 75
            loss_info  = trainer.update(input_data)
            if (iterations + 1) % config['log_iter'] == 0:
                content = f"Iteration: {(iterations+1):08d}/{max_iter:08d}"
                for key, val in loss_info.items():
                    content += f', {key}={val:.6f}'
                logger.info(content)

            #  Check loss in validation set
            if (iterations + 1) % config['validation_iter'] == 0:
                logger.info(f'Start Validation...')
                for val_it, val_input_data in enumerate(val_dataset):
                    val_input_data = val_input_data.cuda() # bs X T X 75
                    if val_it >= 50:
                        break
                    loss_info = trainer.validate(val_input_data)
                    content = f"Val Iteration: {(iterations+1):08d}/{max_iter:08d}"
                    for key, val in loss_info.items():
                        content += f', {key}={val:.6f}'
                    logger.info(content)
                    torch.cuda.empty_cache()
            
            # Visulization
            if (iterations + 1) % config['visualize_iter'] == 0:
                logger.info(f'Iteration: {(iterations+1):08d}/{max_iter:08d}, start visulization...')

                vis_rot_pos_dir = os.path.join(image_directory, str((iterations+1)), 'vis_rot_pos')
                os.makedirs(vis_rot_pos_dir, exist_ok=True)

                for test_it, test_input_data in enumerate(test_dataset):
                    test_input_data = test_input_data.cuda()
                    if test_it >= 10:
                        break
                    concat_seq_cmp = trainer.visualize(test_input_data)

                    # save concat_seq_cmp
                    dst_vis_rot_pos_path = os.path.join(vis_rot_pos_dir, f'{test_it}.npy')
                    np.save(dst_vis_rot_pos_path, concat_seq_cmp.data.cpu().numpy())
                    logger.info(f'{dst_vis_rot_pos_path} saved')
                    torch.cuda.empty_cache()
                
                # start show3d process
                p = Process(
                    target=show3d_multi_refined_rot_pos,
                    args=(config, vis_rot_pos_dir, image_directory, iterations, 'vis_gif', logger)
                )
                p.start()

            # pred root v refineed vibe
            if (iterations + 1) % config['pred_vibe_iter'] == 0:
                logger.info(f'Start pred vibe...')
                vibe_data_dir = config['refined_vibe_data_dir']
                video_list = config['video_list']

                absolute_rot_pos_dir = os.path.join(image_directory, str((iterations+1)), 'absolute_rot_pos_for_vibe')
                os.makedirs(absolute_rot_pos_dir, exist_ok=True)

                absolute_trans_dir = os.path.join(image_directory, str((iterations+1)), 'absolute_trans_for_vibe')
                os.makedirs(absolute_trans_dir, exist_ok=True)

                for video_name in video_list:
                    video_basename = os.path.splitext(video_name)[0]
                    vibe_data_path = os.path.join(vibe_data_dir, f'refined_{video_basename}_rot_mat.npy')
                    absolute_rot_pos, absolute_trans = trainer.refine_vibe(vibe_data_path=vibe_data_path)
                    absolute_rot_pos = absolute_rot_pos.unsqueeze(0) # 1 X T X 24 X 3
                    # save absolute pos for vis
                    dst_path = os.path.join(absolute_rot_pos_dir, f'{video_basename}.npy')
                    np.save(dst_path, absolute_rot_pos.data.cpu().numpy())
                    logger.info(f'{dst_vis_rot_pos_path} saved')

                    # save absolute trans
                    dst_path = os.path.join(absolute_trans_dir, f'{video_basename}.npy')
                    np.save(dst_path, absolute_trans.data.cpu().numpy())
                    logger.info(f'{dst_path} saved')
                    torch.cuda.empty_cache()
                
                # start show3d process
                p = Process(
                    target=show3d_multi_refined_rot_pos,
                    args=(config, absolute_rot_pos_dir, image_directory, iterations, 'absolute_gif', logger)
                )
                p.start()


            if (iterations + 1) % config['log_iter'] == 0:
                write_loss(iterations, trainer, train_writer)

            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                trainer.save(checkpoint_directory, iterations)
                logger.info(f'Iteration: {(iterations+1):08d}/{max_iter:08d}, Saved model')

            iterations += 1
            if iterations >= max_iter:
                logger.info(f'Iteration: {(iterations+1):08d}/{max_iter:08d}, Finish!')
                sys.exit(0)
            trainer.update_learning_rate()