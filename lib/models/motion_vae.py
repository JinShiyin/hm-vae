'''
Description: 
Author: JinShiyin
Email: shiyinjin@foxmail.com
Date: 2022-04-26 21:45:40
'''
import torch.nn as nn
from dotmap import DotMap
from skeleton import get_edges
from lib.models.base_models import Encoder, Decoder


class MotionVAE(nn.Module):
    def __init__(self, hp):
        super(MotionVAE, self).__init__()

        self.hp = hp
        parent_json = hp['parent_json_file']
        # a list with 23(n_joints-1) elements, each elements represents a pair (parent_idx, idx)
        edges = get_edges(parent_json)
        self.enc = Encoder(hp, edges)
        enc_info = DotMap({
            'channel_list': self.enc.channel_list,
            'topologies': self.enc.topologies,
            'channel_base': self.enc.channel_base,
            'pooling_list': self.enc.pooling_list,
            'edge_num': self.enc.edge_num
        })
        self.dec = Decoder(hp, enc_info)

    def forward(self, seq_rot_6d):
        '''
        description: 
        param {*} self
        param {*} seq_rot_6d: bs X T X (24*6)
        return {*} rec_seq_rot_6d: bs X T X (24*6)
        '''
        offset = None
        seq_rot_6d = seq_rot_6d.transpose(0, 2, 1)  # bs X (24*6) X T
        latent = self.enc(seq_rot_6d, offset)
        rec_seq_rot_6d = self.dec(latent, offset)  # bs X (24*6) X T
        rec_seq_rot_6d = rec_seq_rot_6d.transpose(0, 2, 1)  # bs X T X (24*6)
        return rec_seq_rot_6d
