'''
Description: 
Author: JinShiyin
Email: shiyinjin@foxmail.com
Date: 2022-05-01 14:06:51
'''
'''
Description: 
Author: JinShiyin
Email: shiyinjin@foxmail.com
Date: 2022-04-27 14:12:47
'''
import os
import torch
import joblib
import numpy as np
import torch.nn as nn
import torchgeometry as tgm
from lib.models.trajectory_model import TrajectoryModel
from lib.models.fk_layer import ForwardKinematicsLayer
from lib.utils.common_utils import get_model_list, get_scheduler, weights_init
from lib.utils.conversion_utils import amass_pose_to_smpl_pose, convert_to_input, get_all_contact_label, smpl_pose_to_amass_pose
from lib.utils.rotation_utils import hmvae_rot6d_to_rotmat, rotmat_to_rot6d


class TrajectoryModelTrainer(nn.Module):
    def __init__(self, cfg, logger):
        super(TrajectoryModelTrainer, self).__init__()
        self.cfg = cfg
        self.logger = logger
        self.model = TrajectoryModel(cfg)
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
        self.loss_rec_absolute_rot_pos = torch.zeros(1).cuda()
        self.loss_rec_root_v = torch.zeros(1).cuda()
        self.loss_contact = torch.zeros(1).cuda()
        self.loss_rec_total = torch.zeros(1).cuda()
    
    def forward(self, input):
        '''
        description: 
        param {*} self
        param {*} input: bs X T X (24*n_dim), n_dim can be 3 or 3+2
        return {*}
        '''
        bs, timesteps, _ = input.size() # bs X T X (24*n_dim)
        pred_root_v = self.model(input) # bs X T X 3
        input = input.view(bs, timesteps, 24, -1)
        rot_pos = input[:, :, :, :3] # bs X T X 24 X 3
        absolute_pred_rot_pos = self.relative_to_absolute_rot_pos(rot_pos, pred_root_v) # bs X T X 24 X 3
        return pred_root_v, absolute_pred_rot_pos
    
    def relative_to_absolute_rot_pos(self, rot_pos, root_v):
        '''
        description: 
        param {*} self
        param {*} rot_pos: bs X T X 24 X 3
        param {*} root_v: bs X T X 3
        return {*} absolute_rot_pos: bs X T X 24 X 3
        '''
        # pose_data: T X bs X 24 X 3, root are origin
        # root_v_data: T X bs X 3, each timestep t for root represents the relative translation with respect to previous timestep, default is normalized value! 
        rot_pos = rot_pos.transpose(0, 1) # T X bs X 24 X 3
        root_v = root_v.transpose(0,1) # T X bs X 3
        tmp_root_v = root_v.clone() # T X bs X 24 X 3    

        timesteps, bs, _, _ = rot_pos.size()
        absolute_rot_pos = rot_pos.clone()
        root_trans = torch.zeros(bs, 3).cuda() # bs X 3
        for t_idx in range(1, timesteps):
            root_trans += tmp_root_v[t_idx, :, :] # bs X 3
            absolute_rot_pos[t_idx, :, :, :] += root_trans[:, None, :] # bs X 24 X 3  
        absolute_rot_pos = absolute_rot_pos.transpose(0, 1) # bs X T X 24 X 3

        return absolute_rot_pos # bs X T X 24 X 3
        
    def l2_criterion(self, pred, gt):
        assert pred.size() == gt.size() # If the two have different dimensions, there would be weird issues!
        loss = (pred-gt)**2
        return loss.mean()

    def standardize_data(self, ori_data):
        # ori_data: T X n_dim
        mean_val = self.mean_std_data[[0], :] # 1 X n_dim
        std_val = self.mean_std_data[[1], :] # 1 X n_dim
        dest_data = (ori_data - mean_val)/(std_val+1e-8) # T X n_dim
        return dest_data

    def standardize_data_specify_dim(self, ori_data, start_idx, end_idx):
        # ori_data: T X n_dim
        mean_val = self.mean_std_data[[0], start_idx:end_idx] # 1 X n_dim
        std_val = self.mean_std_data[[1], start_idx:end_idx] # 1 X n_dim
        dest_data = (ori_data - mean_val)/(std_val+1e-8) # T X n_dim
        return dest_data

    
    def get_velocity(self, pos):
        '''
        description: 
        param {*} self
        param {*} pos: bs X T X 24 X 3
        return {*}
        '''
        minus = torch.cat([pos[:, [0], :, :], pos[:, :-1, :, :]], dim=1)
        return pos-minus # bs X T X 24 X 3
        

    def update(self, data):
        '''
        description: 
        param {*} self
        param {*} data: bs X T X 75(24*3+3), angle_axis(24*3), trans(3)
        return {*}
        '''
        self.opt.zero_grad()
        seq_rot_6d, seq_rot_mat, seq_rot_pos, seq_trans, seq_linear_v, seq_root_v = convert_to_input(
            data=data,
            fk_layer=self.fk_layer,
            # mean_std_data=self.mean_std_data
        )
        bs, timesteps, _ = seq_rot_pos.size() # bs X T X (24*3)

        contact_label = get_all_contact_label(
            rot_pos=seq_rot_pos,
            trans=seq_trans,
            pos_height_thresh=self.cfg['pos_height_thresh'],
            velocity_thresh=self.cfg['velocity_thresh']
        ) # bs X T X 24
        contact_label = contact_label.unsqueeze(-1) # bs X T X 24 X 1
        input_data = torch.cat([seq_rot_pos.view(bs, timesteps, -1, 3), contact_label], dim=-1).contiguous().view(bs, timesteps, -1) # bs X T X (24*4)
        
        pred_root_v, absolute_pred_rot_pos = self.forward(input_data)
        # bs X T X 3, bs X T X 24 X 3
        absolute_rot_pos = self.relative_to_absolute_rot_pos(seq_rot_pos.view(bs, timesteps, 24, -1), seq_root_v) # bs X T X 24 X 3

        pred_velocity = self.get_velocity(absolute_pred_rot_pos) # bs X T X 24 X 3
        pred_velocity = contact_label * pred_velocity # bs X T X 24 X 3, set the no contact to zero
        
        self.loss_rec_root_v = self.l2_criterion(pred_root_v, seq_root_v)
        self.loss_rec_absolute_rot_pos = self.l2_criterion(absolute_pred_rot_pos, absolute_rot_pos)
        self.loss_contact = (pred_velocity**2).mean()

        self.loss_rec_total = self.cfg['rec_root_v_w'] * self.loss_rec_root_v + \
                              self.cfg['rec_absolute_rot_pos_w'] * self.loss_rec_absolute_rot_pos +\
                              self.cfg['contact_w'] * self.loss_contact
        self.loss_rec_total.backward()
        self.opt.step()

        info_dict = {
            'loss_rec_total': self.loss_rec_total.item(),
            'loss_rec_root_v': self.loss_rec_root_v.item(),
            'loss_contact': self.loss_contact.item(),
            'loss_rec_absolute_rot_pos': self.loss_rec_absolute_rot_pos.item(),
        }
        return info_dict
    
    def validate(self, data):
        '''
        description: 
        param {*} self
        param {*} data: bs X T X 75(24*3+3), angle_axis(24*3), trans(3)
        return {*}
        '''
        self.eval()
        with torch.no_grad():
            seq_rot_6d, seq_rot_mat, seq_rot_pos, seq_trans, seq_linear_v, seq_root_v = convert_to_input(
                data=data,
                fk_layer=self.fk_layer,
                # mean_std_data=self.mean_std_data
            )
            
            bs, timesteps, _ = seq_rot_pos.size() # bs X T X (24*3)
        
            contact_label = get_all_contact_label(
                rot_pos=seq_rot_pos,
                trans=seq_trans,
                pos_height_thresh=self.cfg['pos_height_thresh'],
                velocity_thresh=self.cfg['velocity_thresh']
            ) # bs X T X 24
            contact_label = contact_label.unsqueeze(-1) # bs X T X 24 X 1
            input_data = torch.cat([seq_rot_pos.view(bs, timesteps, -1, 3), contact_label], dim=-1).contiguous().view(bs, timesteps, -1) # bs X T X (24*4)
            
            pred_root_v, absolute_pred_rot_pos = self.forward(input_data)
            # bs X T X 3, bs X T X 24 X 3
            absolute_rot_pos = self.relative_to_absolute_rot_pos(seq_rot_pos.view(bs, timesteps, 24, -1), seq_root_v) # bs X T X 24 X 3

            pred_velocity = self.get_velocity(absolute_pred_rot_pos) # bs X T X 24 X 3
            pred_velocity = contact_label * pred_velocity # bs X T X 24 X 3, set the no contact to zero
            
            self.loss_rec_root_v = self.l2_criterion(pred_root_v, seq_root_v)
            self.loss_rec_absolute_rot_pos = self.l2_criterion(absolute_pred_rot_pos, absolute_rot_pos)
            self.loss_contact = (pred_velocity**2).mean()

            self.loss_rec_total = self.cfg['rec_root_v_w'] * self.loss_rec_root_v + \
                                self.cfg['rec_absolute_rot_pos_w'] * self.loss_rec_absolute_rot_pos +\
                                self.cfg['contact_w'] * self.loss_contact

            info_dict = {
                'loss_rec_total': self.loss_rec_total.item(),
                'loss_rec_root_v': self.loss_rec_root_v.item(),
                'loss_contact': self.loss_contact.item(),
                'loss_rec_absolute_rot_pos': self.loss_rec_absolute_rot_pos.item(),
            }
            return info_dict
    
    def visualize(self, data):
        '''
        description: 
        param {*} self
        param {*} data: bs X T X 75(24*3+3), angle_axis(24*3), trans(3)
        return {*}
        '''
        self.eval()
        with torch.no_grad():
            seq_rot_6d, seq_rot_mat, seq_rot_pos, seq_trans, seq_linear_v, seq_root_v = convert_to_input(
                data=data,
                fk_layer=self.fk_layer,
                # mean_std_data=self.mean_std_data
            )
            
            bs, timesteps, _ = seq_rot_pos.size() # bs X T X (24*3)
        
            contact_label = get_all_contact_label(
                rot_pos=seq_rot_pos,
                trans=seq_trans,
                pos_height_thresh=self.cfg['pos_height_thresh'],
                velocity_thresh=self.cfg['velocity_thresh']
            ) # bs X T X 24
            contact_label = contact_label.unsqueeze(-1) # bs X T X 24 X 1
            input_data = torch.cat([seq_rot_pos.view(bs, timesteps, -1, 3), contact_label], dim=-1).contiguous().view(bs, timesteps, -1) # bs X T X (24*4)
            
            pred_root_v, absolute_pred_rot_pos = self.forward(input_data)
            # bs X T X 3, bs X T X 24 X 3
            absolute_rot_pos = self.relative_to_absolute_rot_pos(seq_rot_pos.view(bs, timesteps, 24, -1), seq_root_v) # bs X T X 24 X 3
            concat_seq_cmp = torch.cat([absolute_rot_pos[[0], :, :, :], absolute_pred_rot_pos[[0], :, :, :]], dim=0) # 2 X T X 24 X 3
        self.train()

        return concat_seq_cmp
    
    def get_absolute_trans(self, start_pos, root_v):
        '''
        description: 
        param {*} self
        param {*} start_pos: (3,)
        param {*} root_v: T X 3
        return {*} absolute_trans: T X 3
        '''
        timesteps, _ = root_v.size()
        absolute_trans = torch.zeros(timesteps, 3).cuda()
        absolute_trans[0] = start_pos
        for i in range(1, timesteps):
            absolute_trans[i] = absolute_trans[i-1] + root_v[i]
        return absolute_trans
    

    def refine_vibe(self, vibe_data_path):
        self.eval()
        with torch.no_grad():
            seq_rot_mat = np.load(vibe_data_path) # T X 24 X 3 X 3, smpl format
            seq_rot_mat = torch.from_numpy(seq_rot_mat).float().cuda() # T X 24 X 3 X 3
            seq_rot_mat = smpl_pose_to_amass_pose(seq_rot_mat) # T X 24 X 3 X 3, amass format
            self.logger.info('Start fk')
            seq_rot_pos = self.fk_layer(seq_rot_mat) # T X 24 X 3
            timesteps, _, _ = seq_rot_pos.size()

            zeros = torch.zeros(timesteps, 24, 1).cuda()

            trans = torch.zeros(timesteps, 3).cuda() # absolute trans
            # Process sequence with our model in sliding window fashion, use the centering frame strategy
            window_size = self.cfg['train_seq_len'] # 64
            center_frame_start_idx = window_size // 2 - 1
            center_frame_end_idx = window_size // 2 - 1
            for t_idx in range(0, timesteps-window_size+1, 1):
                # pred_root_v = self.model(seq_rot_pos[t_idx:t_idx+window_size, :, :].unsqueeze(0).contiguous().view(1, window_size, -1)) # 1 X window_size X 3

                input_data = torch.cat([seq_rot_pos[t_idx:t_idx+window_size, :, :], zeros[t_idx:t_idx+window_size, :, :]], dim=-1)
                input_data = input_data.unsqueeze(0).contiguous().view(1, window_size, -1) # 1 X window_size X 4
                pred_root_v = self.model(input_data)

                if t_idx == 0:
                    start_pos = trans[t_idx]
                    absolute_trans = self.get_absolute_trans(start_pos, pred_root_v.squeeze(0)) # window_size X 3
                    # The beginning part, we take all the frames before center
                    trans[:center_frame_end_idx+1, :] = absolute_trans[:center_frame_end_idx+1, :] 
                elif t_idx == timesteps-window_size:
                    start_pos = trans[t_idx]
                    # Handle the last window in the end, take all the frames after center_start to make the videl length same as input
                    absolute_trans = self.get_absolute_trans(start_pos, pred_root_v.squeeze(0)) # window_size X 3
                    trans[t_idx+center_frame_start_idx:, :] = absolute_trans[center_frame_start_idx:, :]
                else:
                    start_pos = trans[t_idx]
                    absolute_trans = self.get_absolute_trans(start_pos, pred_root_v.squeeze(0)) # window_size X 3
                    trans[t_idx+center_frame_end_idx, :] = absolute_trans[center_frame_end_idx, :]
            
            absolute_rot_pos = seq_rot_pos + trans.unsqueeze(1) # T X 24 X 3            
        
        self.train()

        return absolute_rot_pos, trans


    def update_learning_rate(self):
        if self.scheduler is not None:
            self.scheduler.step()
    
    def resume(self, checkpoint_dir):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.model.load_state_dict(state_dict)

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
        torch.save(self.model.state_dict(), gen_name)
        torch.save(self.opt.state_dict(), opt_name)
    





        


