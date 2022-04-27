'''
Description: 
Author: JinShiyin
Email: shiyinjin@foxmail.com
Date: 2022-04-26 21:56:17
'''
import os
import sys
import torch
import shutil
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

from lib.utils.logs import init_logger
from lib.utils.common_utils import setup_seed, write_loss
from utils_motion_vae import get_train_loaders_all_data_seq
from utils_common import show3Dpose_animation_multiple
from lib.utils.common_utils import init_config, create_sub_folders
from lib.trainer.motion_vae_trainer import MotionVAETrainer


if __name__ == '__main__':
    # Enable auto-tuner to find the best algorithm to use for your hardware.
    cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='', help='configuration file for training and testing')
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Load experiment setting
    config = init_config(args.config)
    max_iter = config['max_iter']

    # Setup logger and output folders
    output_dir = config['output_dir']
    logger = init_logger(log_name='MotionVAE', log_dir=output_dir)
    log_dir, checkpoint_directory, image_directory = create_sub_folders(output_dir)
    if not args.resume:
        shutil.copyfile(args.config, os.path.join(output_dir, 'config.yaml'))
    train_writer = SummaryWriter(log_dir)

    trainer = MotionVAETrainer(config, logger)
    trainer.cuda()
    iterations = trainer.resume(checkpoint_directory) if args.resume else 0

    data_loaders = get_train_loaders_all_data_seq(config)
    train_loader = data_loaders[0]
    val_loader = data_loaders[1]
    test_loader = data_loaders[2]

    epoch = 0
    while True:
        epoch += 1
        train_dataset = train_loader
        val_dataset = val_loader
        test_dataset = test_loader
        for it, input_data in enumerate(train_dataset):
            for i in range(len(input_data)):
                input_data[i] = input_data[i].float().cuda()
            loss_info  = trainer.update(input_data)
            if (iterations + 1) % config['log_iter'] == 0:
                content = f"Iteration: {(iterations+1):08d}/{max_iter:08d}"
                for key, val in loss_info.items():
                    content += f', {key}={val:.6f}'
                logger.info(content)

            # Check loss in validation set
            if (iterations + 1) % config['validation_iter'] == 0:
                for val_it, val_input_data in enumerate(val_dataset):
                    for i in range(len(val_input_data)):
                        val_input_data[i] = val_input_data[i].float().cuda()
                    if val_it >= 50:
                        break
                    loss_info = trainer.validate(val_input_data)
                    content = f"Val Iteration: {(iterations+1):08d}/{max_iter:08d}"
                    for key, val in loss_info.items():
                        content += f', {key}={val:.6f}'
                    logger.info(content)
            
            # Visulization
            if (iterations + 1) % config['visualize_iter'] == 0:
                logger.info(f'Iteration: {(iterations+1):08d}/{max_iter:08d}, start visulization...')
                for test_it, test_input_data in enumerate(test_dataset):
                    for i in range(len(test_input_data)):
                        test_input_data[i] = test_input_data[i].float().cuda()
                    if test_it >= 10:
                        break
                    seq_rot_6d, seq_rot_mat, seq_rot_pos, _, _, _, _ = test_input_data
                    rec_seq_rot_6d, rec_seq_rot_mat, rec_seq_rot_pos = trainer.visualize(seq_rot_6d)
                    bs, timesteps, _ = seq_rot_6d.size()
                    seq_rot_pos = seq_rot_pos[0].view(1, timesteps, 24, 3)
                    rec_seq_rot_pos = rec_seq_rot_pos[0].view(1, timesteps, 24, 3)
                    concat_seq_cmp = torch.cat([seq_rot_pos, rec_seq_rot_pos], dim=0)
                    show3Dpose_animation_multiple(
                        channels=concat_seq_cmp.data.cpu().numpy(), 
                        image_directory=image_directory, 
                        iterations=(iterations+1), 
                        tag="reconstruct", 
                        bs_idx=test_it, 
                        use_joint12=False, 
                        use_amass=True, 
                        use_lafan=False, 
                        make_translation=False
                    )

            if (iterations + 1) % config['log_iter'] == 0:
                write_loss(iterations, trainer, train_writer)

            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                trainer.save(checkpoint_directory, iterations)
                logger.info(f'Iteration: {(iterations+1):08d}/{max_iter:08d}, Saved model')

            iterations += 1
            if iterations >= max_iter:
                logger.info(f'Iteration: {(iterations+1):08d}/{max_iter:08d}, Finish!')
                sys.exit(0)
