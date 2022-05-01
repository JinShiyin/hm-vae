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
from lib.utils.conversion_utils import amass_pose_to_smpl_pose, convert_to_input, smpl_pose_to_amass_pose
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
            mean_std_data=self.mean_std_data
        )
        bs, timesteps, _ = seq_rot_pos.size() # bs X T X (24*3)
        
        pred_root_v, absolute_pred_rot_pos = self.forward(seq_rot_pos)
        # bs X T X 3, bs X T X 24 X 3
        absolute_rot_pos = self.relative_to_absolute_rot_pos(seq_rot_pos.view(bs, timesteps, 24, -1), seq_root_v) # bs X T X 24 X 3

        self.loss_rec_root_v = self.l2_criterion(pred_root_v, seq_root_v)
        self.loss_rec_absolute_rot_pos = self.l2_criterion(absolute_pred_rot_pos, absolute_rot_pos)

        self.loss_rec_total = self.cfg['rec_root_v_w'] * self.loss_rec_root_v + \
                           self.cfg['rec_absolute_rot_pos'] * self.loss_rec_absolute_rot_pos
        self.loss_rec_total.backward()
        self.opt.step()

        info_dict = {
            'loss_rec_total': self.loss_rec_total.item(),
            'loss_rec_root_v': self.loss_rec_root_v.item(),
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
                mean_std_data=self.mean_std_data
            )
            
            bs, timesteps, _ = seq_rot_pos.size() # bs X T X (24*3)
        
            pred_root_v, absolute_pred_rot_pos = self.forward(seq_rot_pos)
            # bs X T X 3, bs X T X 24 X 3
            absolute_rot_pos = self.relative_to_absolute_rot_pos(seq_rot_pos.view(bs, timesteps, 24, -1), seq_root_v) # bs X T X 24 X 3

            self.loss_rec_root_v = self.l2_criterion(pred_root_v, seq_root_v)
            self.loss_rec_absolute_rot_pos = self.l2_criterion(absolute_pred_rot_pos, absolute_rot_pos)

            self.loss_rec_total = self.cfg['rec_root_v_w'] * self.loss_rec_root_v + \
                            self.cfg['rec_absolute_rot_pos'] * self.loss_rec_absolute_rot_pos
            self.loss_rec_total.backward()
            self.opt.step()

            info_dict = {
                'loss_rec_total': self.loss_rec_total.item(),
                'loss_rec_root_v': self.loss_rec_root_v.item(),
                'loss_rec_absolute_rot_pos': self.loss_rec_absolute_rot_pos.item(),
            }
        self.train()
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
                mean_std_data=self.mean_std_data
            )
            
            bs, timesteps, _ = seq_rot_pos.size() # bs X T X (24*3)
        
            pred_root_v, absolute_pred_rot_pos = self.forward(seq_rot_pos)
            # bs X T X 3, bs X T X 24 X 3
            absolute_rot_pos = self.relative_to_absolute_rot_pos(seq_rot_pos.view(bs, timesteps, 24, -1), seq_root_v) # bs X T X 24 X 3
            concat_seq_cmp = torch.cat([absolute_rot_pos[[0], :, :, :], absolute_pred_rot_pos[[0], :, :, :]], dim=0) # 2 X T X 24 X 3
        self.train()

        return concat_seq_cmp
    

    def refine_vibe(self, vibe_data_path):
        self.eval()
        with torch.no_grad():
            pred_theta_data = joblib.load(vibe_data_path)
            pred_theta_data = pred_theta_data[1]['pose'] # T X 72
            timesteps, _ = pred_theta_data.shape 

            # Process predicted results from other methods as input to encoder
            pred_aa_data = torch.from_numpy(pred_theta_data).float().cuda() # T X 72
            pred_rot_mat = tgm.angle_axis_to_rotation_matrix(pred_aa_data.view(-1, 3))[:, :3, :3] # (T*24) X 3 X 3
            pred_rot_mat = pred_rot_mat.view(-1, 24, 3, 3) # T X 24 X 3 X 3
            pred_rot_mat = smpl_pose_to_amass_pose(pred_rot_mat) # T X 24 X 3 X 3
            pred_rot_6d = rotmat_to_rot6d(pred_rot_mat) # (T*24) X 6
            pred_rot_6d = pred_rot_6d.view(-1, 24, 6).unsqueeze(0) # 1 X T X 24 X 6

            # Process sequence with our model in sliding window fashion, use the centering frame strategy
            window_size = self.cfg['train_seq_len'] # 64
            center_frame_start_idx = self.cfg['train_seq_len'] // 2 - 1
            center_frame_end_idx = self.cfg['train_seq_len'] // 2 - 1
            # Options: 7, 7; 
            overlap_len = window_size - (center_frame_end_idx-center_frame_start_idx+1)
            stride = window_size - overlap_len
            our_pred_6d_out_seq = None # T X 24 X 6
            

            for t_idx in range(0, timesteps-window_size+1, stride):
                curr_encoder_input = pred_rot_6d[:, t_idx:t_idx+window_size, :, :].cuda() # bs(1) X 16 X 24 X 6
                our_rec_6d_out = self.model(curr_encoder_input.view(1, window_size, -1)) # 1 X 64 X (24*6)
                our_rec_6d_out = our_rec_6d_out.view(1, window_size, 24, 6)

                if t_idx == 0:
                    # The beginning part, we take all the frames before center
                    our_pred_6d_out_seq = our_rec_6d_out.squeeze(0)[:center_frame_end_idx+1, :, :] 
                elif t_idx == timesteps-window_size:
                    # Handle the last window in the end, take all the frames after center_start to make the videl length same as input
                    our_pred_6d_out_seq = torch.cat((our_pred_6d_out_seq, \
                        our_rec_6d_out[0, center_frame_start_idx:, :, :]), dim=0)
                else:
                    our_pred_6d_out_seq = torch.cat((our_pred_6d_out_seq, \
                        our_rec_6d_out[0, center_frame_start_idx:center_frame_end_idx+1, :, :]), dim=0)

            self.logger.info(f'start fk...')
            # Use same skeleton for visualization
            pred_fk_pose = self.fk_layer(pred_rot_6d.squeeze(0)) # T X 24 X 3, vibe
            our_fk_pose = self.fk_layer(our_pred_6d_out_seq) # T X 24 X 3, hmvae
            our_fk_pose[:, :, 0] += 1

            concat_seq_cmp = torch.cat((pred_fk_pose[None, :, :, :], our_fk_pose[None, :, :, :]), dim=0) # 2 X T X 24 X 3

            # from amass to smpl
            our_pred_rot_mat = hmvae_rot6d_to_rotmat(our_pred_6d_out_seq.view(-1, 6)) # (T*24) X 3 X 3(our_pred_6d_out_seq)
            our_pred_rot_mat = our_pred_rot_mat.view(-1, 24, 3, 3) # T X 24 X 3 X 3
            our_pred_rot_mat = amass_pose_to_smpl_pose(our_pred_rot_mat) # T X 24 X 3 X 3
            pred_rot_mat = pred_rot_mat.view(-1, 24, 3, 3) # T X 24 X 3 X 3
            pred_rot_mat = amass_pose_to_smpl_pose(pred_rot_mat) # T X 24 X 3 X 3
        self.train()

        return concat_seq_cmp, our_pred_rot_mat, pred_rot_mat

    
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
    





        


