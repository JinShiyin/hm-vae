'''
Description: 
Author: JinShiyin
Email: shiyinjin@foxmail.com
Date: 2022-04-27 14:46:25
'''
import math
import os
import time
import torch
import yaml
import random
from torch.nn import init
from torch.optim import lr_scheduler
import numpy as np


def setup_seed(seed):
    if seed >= 0:
        print(f'Seed value for the experiment {seed}')
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(seed)
    else:
        print(f'Seed < 0, not set')


def create_sub_folders(output_path, if_test=False):
    log_dir = os.path.join(output_path, 'logs')
    checkpoint_dir = os.path.join(output_path, 'checkpoints')
    image_dir = os.path.join(output_path, 'images')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    if if_test:
        test_dir = os.path.join(output_path, 'test')
        os.makedirs(test_dir, exist_ok=True)
        return log_dir, checkpoint_dir, image_dir, test_dir
    else:
        return log_dir, checkpoint_dir, image_dir


def init_config(config_path):
    config = get_config(config_path)
    exp_name = config['exp_name']
    output_dir = config['output_dir']
    time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    config['output_dir'] = os.path.join(output_dir, exp_name, f'{time_stamp}-{exp_name}')
    return config


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None  # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer)
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and (
        'loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)
