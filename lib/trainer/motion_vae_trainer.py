'''
Description: 
Author: JinShiyin
Email: shiyinjin@foxmail.com
Date: 2022-04-27 14:12:47
'''
import os
import torch
import torch.nn as nn
import numpy as np
from fk_layer import ForwardKinematicsLayer
from lib.models.motion_vae import MotionVAE
from lib.utils.common_utils import get_model_list, get_scheduler, weights_init
from lib.utils.rotation_utils import hmvae_rot6d_to_rotmat


class MotionVAETrainer(nn.Module):
    def __init__(self, cfg, logger):
        super(MotionVAETrainer, self).__init__()
        self.cfg = cfg
        self.logger = logger
        self.model = MotionVAE(cfg)
        self.fk_layer = ForwardKinematicsLayer()

        self.logger.info('Fix the parameters in fk layer')
        for param in self.fk_layer.parameters():
            param.requires_grad = False
        
        self.mean_std_data = torch.from_numpy(np.load(cfg['mean_std_path'])).float().cuda() # 2 X 579

        # set optimizer and scheduler
        params = list(self.model.parameters())
        self.opt = torch.optim.Adam([p for p in params if p.requires_grad], lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        self.scheduler = get_scheduler(self.opt, cfg)

        # init model
        self.model.apply(weights_init(cfg['init']))

        # init loss
        self.loss_rec_rot_6d = torch.zeros(1).cuda()
        self.loss_rec_rot_mat = torch.zeros(1).cuda()
        self.loss_rec_rot_pos = torch.zeros(1).cuda()
        self.loss_rec_joint_pos = torch.zeros(1).cuda()
        self.loss_rec_linear_v = torch.zeros(1).cuda()
        self.loss_rec_angular_v = torch.zeros(1).cuda()
        self.loss_rec_root_v = torch.zeros(1).cuda()
        self.loss_rec_total = torch.zeros(1).cuda()
    
    def forward(self, seq_rot_6d):
        '''
        description: 
        param {*} self
        param {*} seq_rot_6d: bs X T X (24*6)
        return {*}
        '''
        bs, timesteps, _ = seq_rot_6d.size() # bs X T X (24*6)
        n_joints = self.cfg['n_joints']
        # get rec_rot_6d
        rec_seq_rot_6d = self.model(seq_rot_6d) # bs X T X (24*6)
        rec_seq_rot_mat = hmvae_rot6d_to_rotmat(rec_seq_rot_6d.view(bs, timesteps, n_joints, 6)) # bs X T X 24 X 3 X 3
        rec_seq_rot_mat = rec_seq_rot_mat.view(-1, n_joints, 3, 3) # (bs*T) X 24 X 3 X 3
        rec_seq_rot_pos = self.fk_layer(rec_seq_rot_mat) # (bs*T) X 24 X 3

        rec_seq_rot_mat = rec_seq_rot_mat.contiguous().view(bs, timesteps, -1) # bs X T X (24*3*3)
        rec_seq_rot_pos = rec_seq_rot_pos.contiguous().view(bs, timesteps, -1) # bs X T X (24*3)

        # normalize rec_seq_rot_pos to get rec_seq_joint_pos
        start_idx = 24*6+24*3*3+24*3
        rec_seq_joint_pos = self.standardize_data_specify_dim(
            ori_data=rec_seq_rot_pos.view(-1, n_joints*3),
            start_idx=start_idx,
            end_idx=start_idx+24*3
        )
        rec_seq_joint_pos = rec_seq_joint_pos.view(bs, timesteps, -1)

        # diff rec_seq_rot_pos and normalize to cal linear_v
        minus_rec_seq_rot_pos = rec_seq_rot_pos[:, :-1, :] # bs X T-1 X (24*3)
        minus_rec_seq_rot_pos = torch.cat([rec_seq_rot_pos[:, [0], :], minus_rec_seq_rot_pos], dim=1) # bs X T X (24*3)
        rec_seq_linear_v = rec_seq_rot_pos - minus_rec_seq_rot_pos # bs X T X (24*3)

        start_idx = 24*6 + 24*3*3 + 24*3 + 24*3
        rec_seq_linear_v = self.standardize_data_specify_dim(
            ori_data=rec_seq_linear_v.view(-1, n_joints*3),
            start_idx=start_idx,
            end_idx=start_idx+24*3
        )
        rec_seq_linear_v = rec_seq_linear_v.view(bs, timesteps, -1)

        return rec_seq_rot_6d, rec_seq_rot_mat, rec_seq_rot_pos, rec_seq_joint_pos, rec_seq_linear_v
        
    def l2_criterion(self, pred, gt):
        assert pred.size() == gt.size() # If the two have different dimensions, there would be weird issues!
        loss = (pred-gt)**2
        return loss.mean()

    def standardize_data(self, ori_data):
        # ori_data: T X n_dim
        mean_val = self.mean_std_data[[0], :] # 1 X n_dim
        std_val = self.mean_std_data[[1], :] # 1 X n_dim
        dest_data = (ori_data - mean_val)/std_val # T X n_dim
        return dest_data

    def standardize_data_specify_dim(self, ori_data, start_idx, end_idx):
        # ori_data: T X n_dim
        mean_val = self.mean_std_data[[0], start_idx:end_idx] # 1 X n_dim
        std_val = self.mean_std_data[[1], start_idx:end_idx] # 1 X n_dim
        dest_data = (ori_data - mean_val)/std_val # T X n_dim
        return dest_data


    def update(self, data):
        self.opt.zero_grad()
        seq_rot_6d, seq_rot_mat, seq_rot_pos, seq_joint_pos, seq_linear_v, seq_angular_v, seq_root_v = data
        
        rec_seq_rot_6d, rec_seq_rot_mat, rec_seq_rot_pos, rec_seq_joint_pos, rec_seq_linear_v = self.forward(seq_rot_6d)

        if self.cfg['rec_joint_pos_w'] != 0:
            self.loss_rec_joint_pos = self.l2_criterion(rec_seq_joint_pos, seq_joint_pos)

        if self.cfg['rec_linear_v_w'] != 0:
            self.loss_rec_linear_v = self.l2_criterion(rec_seq_linear_v, seq_linear_v)
        
        self.loss_rec_rot_6d = self.l2_criterion(rec_seq_rot_6d, seq_rot_6d)
        self.loss_rec_rot_mat = self.l2_criterion(rec_seq_rot_mat, seq_rot_mat)
        self.loss_rec_rot_pos = self.l2_criterion(rec_seq_rot_pos, seq_rot_pos)

        self.loss_rec_total = self.cfg['rec_rot_6d_w'] * self.loss_rec_rot_6d + \
                           self.cfg['rec_rot_mat_w'] * self.loss_rec_rot_mat + \
                           self.cfg['rec_rot_pos_w'] * self.loss_rec_rot_pos + \
                           self.cfg['rec_joint_pos_w'] * self.loss_rec_joint_pos + \
                           self.cfg['rec_linear_v_w'] * self.loss_rec_linear_v
        self.loss_rec_total.backward()
        self.opt.step()

        info_dict = {
            'loss_rec_total': self.loss_rec_total.item(),
            'loss_rec_rot_6d': self.loss_rec_rot_6d.item(),
            'loss_rec_rot_mat': self.loss_rec_rot_mat.item(),
            'loss_rec_rot_pos': self.loss_rec_rot_pos.item(),
            'loss_rec_joint_pos': self.loss_rec_joint_pos.item(),
            'loss_rec_linear_v': self.loss_rec_linear_v.item(),
            'loss_rec_angular_v': self.loss_rec_angular_v.item(),
            'loss_rec_root_v': self.loss_rec_root_v.item(),
        }
        return info_dict
    
    def validate(self, data):
        self.eval()
        with torch.no_grad():
            seq_rot_6d, seq_rot_mat, seq_rot_pos, seq_joint_pos, seq_linear_v, seq_angular_v, seq_root_v = data
            
            rec_seq_rot_6d, rec_seq_rot_mat, rec_seq_rot_pos, rec_seq_joint_pos, rec_seq_linear_v = self.forward(seq_rot_6d)

            if self.cfg['rec_joint_pos_w'] != 0:
                self.loss_rec_joint_pos = self.l2_criterion(rec_seq_joint_pos, seq_joint_pos)

            if self.cfg['rec_linear_v_w'] != 0:
                self.loss_rec_linear_v = self.l2_criterion(rec_seq_linear_v, seq_linear_v)
            
            self.loss_rec_rot_6d = self.l2_criterion(rec_seq_rot_6d, seq_rot_6d)
            self.loss_rec_rot_mat = self.l2_criterion(rec_seq_rot_mat, seq_rot_mat)
            self.loss_rec_rot_pos = self.l2_criterion(rec_seq_rot_pos, seq_rot_pos)

            self.loss_rec_total = self.cfg['rec_rot_6d_w'] * self.loss_rec_rot_6d + \
                            self.cfg['rec_rot_mat_w'] * self.loss_rec_rot_mat + \
                            self.cfg['rec_rot_pos_w'] * self.loss_rec_rot_pos + \
                            self.cfg['rec_joint_pos_w'] * self.loss_rec_joint_pos + \
                            self.cfg['rec_linear_v_w'] * self.loss_rec_linear_v

            info_dict = {
                'loss_rec_total': self.loss_rec_total.item(),
                'loss_rec_rot_6d': self.loss_rec_rot_6d.item(),
                'loss_rec_rot_mat': self.loss_rec_rot_mat.item(),
                'loss_rec_rot_pos': self.loss_rec_rot_pos.item(),
                'loss_rec_joint_pos': self.loss_rec_joint_pos.item(),
                'loss_rec_linear_v': self.loss_rec_linear_v.item(),
                'loss_rec_angular_v': self.loss_rec_angular_v.item(),
                'loss_rec_root_v': self.loss_rec_root_v.item(),
            }
        self.train()
        return info_dict
    
    def visualize(self, seq_rot_6d):
        '''
        description: 
        param {*} self
        param {*} seq_rot_6d: bs X T X (24*6)
        return {*}
        '''
        self.eval()
        with torch.no_grad():
            rec_seq_rot_6d, rec_seq_rot_mat, rec_seq_rot_pos, _, _ = self.forward(seq_rot_6d)
        self.train()
        return rec_seq_rot_6d, rec_seq_rot_mat, rec_seq_rot_pos
    
    def update_learning_rate(self):
        if self.scheduler is not None:
            self.scheduler.step()
    
    def resume(self, checkpoint_dir):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.model.enc.load_state_dict(state_dict['encoder'])
        self.model.dec.load_state_dict(state_dict['decoder'])

        iterations = int(last_model_name[-10:-4])
        # Load optimizers
        last_model_name = get_model_list(checkpoint_dir, "opt")
        state_dict = torch.load(last_model_name)
        self.opt.load_state_dict(state_dict)
        # Reinitilize schedulers
        self.scheduler = get_scheduler(self.opt, self.cfg, iterations)
        self.logger.info('Resume from iteration %d' % iterations)
        del state_dict, last_model_name
        torch.cuda.empty_cache()
        return iterations
    
    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, f'gen_{(iterations + 1):08d}.pth')
        opt_name = os.path.join(snapshot_dir, f'opt_{(iterations + 1):08d}.pth')
        torch.save({'encoder': self.model.enc.state_dict(), 'decoder': self.model.dec.state_dict()}, gen_name)
        torch.save(self.opt.state_dict(), opt_name)
    





        


