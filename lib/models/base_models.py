'''
Description: 
Author: JinShiyin
Email: shiyinjin@foxmail.com
Date: 2022-04-26 21:45:40
'''
from dotmap import DotMap
import torch
import torch.nn as nn
import torchgeometry as tgm
import torch.nn.functional as F
import torch.distributions.multivariate_normal as dist_mn

import my_tools
from fk_layer import ForwardKinematicsLayer
from skeleton import SkeletonUnpool, SkeletonPool, SkeletonConv, find_neighbor, SkeletonLinear, get_edges
from utils_common import show3Dpose_animation, show3Dpose_animation_multiple, show3Dpose_animation_with_mask
from lib.utils.rotation_utils import hmvae_rot6d_to_rotmat, rotmat_to_rot6d
from lib.utils.conversion_utils import smpl_pose_to_amass_pose, amass_pose_to_smpl_pose

class Encoder(nn.Module):
    def __init__(self, args, topology):
        super(Encoder, self).__init__()
        self.topologies = [topology]
      
        self.latent_d = args['latent_d'] # 24

        self.channel_base = [6] # 6D representation 
        self.timestep_list = [args['train_seq_len']]
    
        self.channel_list = []
        self.edge_num = [len(topology)]
        self.pooling_list = []
        self.layers = nn.ModuleList()
        self.args = args
        self.convs = []

        kernel_size = args['kernel_size'] # 15
        padding = (kernel_size - 1) // 2
        bias = True

        for i in range(args['num_layers']): # 4
            self.channel_base.append(self.channel_base[-1] * 2) # 6, 12, 24, 48, 96 # 维度 d 的变化 conv 降T 升d，pool 将edge 不变T与d
          
            if args['train_seq_len'] == 8:
                if i == 0 or i == args['num_layers'] - 1:
                    self.timestep_list.append(self.timestep_list[-1]) # 8, 8, 4, 2, 2
                else:
                    self.timestep_list.append(self.timestep_list[-1]//2)
            elif args['train_seq_len'] == 16:
                if i == 0: # For len = 16
                    self.timestep_list.append(self.timestep_list[-1]) # 16, 16, 8, 4, 2
                else:
                    self.timestep_list.append(self.timestep_list[-1]//2) 
            else:
                self.timestep_list.append(self.timestep_list[-1]//2) # 64, 32, 16, 8, 4
            # print("timestep list:{0}".format(self.timestep_list))

        for i in range(args['num_layers']):
            seq = []
            neighbor_list = find_neighbor(self.topologies[i], args['skeleton_dist']) # 24, 14, 9, 7
            in_channels = self.channel_base[i] * self.edge_num[i] # 6 * 24, 12 * 14, 24 * 9, 48 * 7,
            out_channels = self.channel_base[i+1] * self.edge_num[i] # 12 * 24, 24 * 14, 48 * 9,  96 * 7
         
            if i == 0: self.channel_list.append(in_channels)
            self.channel_list.append(out_channels) # 6*24, 12*14, 24*9, 48*7, 96*7

            for _ in range(args['extra_conv']):
                seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=in_channels,
                                        joint_num=self.edge_num[i], kernel_size=kernel_size, stride=1,
                                        padding=padding, padding_mode=args['padding_mode'], bias=bias))
          
            if args['train_seq_len'] == 8:
                if i == 0 or i == args['num_layers'] - 1:
                    curr_stride = 1
                else:
                    curr_stride = 2
            elif args['train_seq_len'] == 16:
                if i == 0:
                    curr_stride = 1
                else:
                    curr_stride = 2 
            else:
                curr_stride = 2

            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=self.edge_num[i], kernel_size=kernel_size, stride=curr_stride,
                                    padding=padding, padding_mode=args['padding_mode'], bias=bias, add_offset=False,
                                    in_offset_channel=3 * self.channel_base[i] // self.channel_base[0]))
            # print(seq[-1].description)
            # padding_mode=args['padding_mode']
            # in_offset_channel=3 * self.channel_base[i] // self.channel_base[0]
            # print(f"SkeletonConv: neighbor_list={neighbor_list}, in_channels={in_channels}, out_channels={out_channels}, joint_num={self.edge_num[i]}, kernel_size={kernel_size}, stride={curr_stride}, padding={padding}, padding_mode={padding_mode}, bias={bias}, add_offset={False}, in_offset_channel={in_offset_channel}")

            self.convs.append(seq[-1])
            last_pool = True if i == args['num_layers'] - 1 else False
            pool = SkeletonPool(edges=self.topologies[i], pooling_mode=args['skeleton_pool'],
                                channels_per_edge=out_channels // len(neighbor_list), last_pool=last_pool)
            seq.append(pool)
            # print(seq[-1].description)
            seq.append(nn.LeakyReLU(negative_slope=0.2))
            self.layers.append(nn.Sequential(*seq))
    
            self.topologies.append(pool.new_edges)
            self.pooling_list.append(pool.pooling_list)
            self.edge_num.append(len(self.topologies[-1]))
        self.latent_enc_layer = nn.Linear(self.channel_base[-1]*self.timestep_list[-1], self.latent_d)

    def forward(self, input, offset=None):
        '''
        description: 
        param {*} self
        param {*} input: bs X (24*6) X T
        param {*} offset
        return {*}
        '''
        for i, layer in enumerate(self.layers):
            input = layer(input)
        # input = self.layers(input)
        bs, _, compressed_t = input.size() # bs X (k_edges*d) X (T//2^n_layers)
        k_edges = input.shape[1] // self.channel_base[-1] # edge_num, channel_base 存储了 d
        encoder_map_input = input.view(bs, k_edges, -1)
        z_vector = self.latent_enc_layer(encoder_map_input) # bs X k_edges X (2*latent_d)
        return z_vector

class Decoder(nn.Module):
    def __init__(self, args, enc_info):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.unpools = []

        self.latent_d = args['latent_d']

        self.args = args
        self.enc = enc_info
        self.convs = []

        self.hp = args

        self.timestep_list = [args['train_seq_len']]
        for i in range(args['num_layers']):
            if args['train_seq_len'] == 8:
                if i == 0 or i == args['num_layers'] - 1:
                    self.timestep_list.append(self.timestep_list[-1]) # 16, 16, 8, 4, 2(8, 8, 4, 2, 2)
                else:
                    self.timestep_list.append(self.timestep_list[-1]//2) # 64, 32, 16, 8, 4
            elif args['train_seq_len'] == 16:
                if i == 0:
                    self.timestep_list.append(self.timestep_list[-1]) # 16, 16, 8, 4, 2(8, 8, 4, 2)
                else:
                    self.timestep_list.append(self.timestep_list[-1]//2) # 64, 32, 16, 8, 4
            else:
                self.timestep_list.append(self.timestep_list[-1]//2) # 64, 32, 16, 8, 4

        self.timestep_list.reverse() # 4, 8, 16, 32, 64 ( 2, 2, 4, 8, 8; 2, 4, 8, 16, 16)

        kernel_size = args['kernel_size']
        padding = (kernel_size - 1) // 2

        for i in range(args['num_layers']):
            seq = []
            
            in_channels = self.enc.channel_list[args['num_layers'] - i] # 7*96, 9*48, 14*24, 24*12
            # print("in channels:{0}".format(in_channels))  
            out_channels = in_channels // 2 # 7*96->7*24, 9*48->9*12, 14*24->14*6, 24*12->24*6
            
            neighbor_list = find_neighbor(self.enc.topologies[args['num_layers'] - i - 1], args['skeleton_dist']) # 7, 9, 14, 24
            # neighbor_list = find_neighbor(enc.topologies[args['num_layers'] - i], args['skeleton_dist']) # 7, 7, 9, 14

            if i != 0 and i != args['num_layers'] - 1:
                bias = False
            else:
                bias = True
          
            # if i == args['num_layers'] - 1:
            #     latent_decode_linear = nn.Linear(self.shallow_latent_d, enc.channel_base[args['num_layers']-i]*self.timestep_list[i]) # 96*4
            # else:
            #     latent_decode_linear = nn.Linear(self.latent_d, enc.channel_base[args['num_layers']-i]*self.timestep_list[i]) # 24*8, 12*16, 6*32
            # self.latent_dec_layers.append(latent_decode_linear)

            self.latent_dec_layer = nn.Linear(self.latent_d, self.enc.channel_base[-1]*self.timestep_list[0])

            self.unpools.append(SkeletonUnpool(self.enc.pooling_list[args['num_layers'] - i - 1], in_channels // len(neighbor_list)))

            if args['train_seq_len'] == 8:
                if i != args['num_layers'] - 1 and i != 0:
                    seq.append(nn.Upsample(scale_factor=2, mode=args['upsampling'], align_corners=False))
            elif args['train_seq_len'] == 16:
                if i != args['num_layers'] - 1:
                    seq.append(nn.Upsample(scale_factor=2, mode=args['upsampling'], align_corners=False))
            else:
                seq.append(nn.Upsample(scale_factor=2, mode=args['upsampling'], align_corners=False))

            seq.append(self.unpools[-1])
            for _ in range(args['extra_conv']):
                seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=in_channels,
                                        joint_num=self.enc.edge_num[args['num_layers'] - i - 1], kernel_size=kernel_size,
                                        stride=1,
                                        padding=padding, padding_mode=args['padding_mode'], bias=bias))
      
            curr_stride = 1 
           
            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=self.enc.edge_num[args['num_layers'] - i - 1], kernel_size=kernel_size, stride=curr_stride,
                                    padding=padding, padding_mode=args['padding_mode'], bias=bias, add_offset=False,
                                    in_offset_channel=3 * self.enc.channel_base[args['num_layers'] - i - 1] // self.enc.channel_base[0]))
            print(f'in_channels={in_channels}, out_channels={out_channels}')

            self.convs.append(seq[-1])
            if i != args['num_layers'] - 1: seq.append(nn.LeakyReLU(negative_slope=0.2))

            self.layers.append(nn.Sequential(*seq))

    def forward(self, z_vec, offset=None):
        # train_hier_level: 1, 2, 3, 4 (deep to shallow)
        # feat = self.latent_dec_layer(z_vec)
        input = self.latent_dec_layer(z_vec)
        bs = input.size()[0]
        input = input.view(bs, -1, self.timestep_list[0])

        for i, layer in enumerate(self.layers):
            # print(f'i={i}, layer={layer}, in_size={input.size()}')
            input = layer(input)
            # print(f'out_size={input.size()}')
        return input


if __name__ == '__main__':
    from utils_common import get_config
    config_path = 'configs/len64_no_aug_hm_vae.yaml'
    config = get_config(config_path)

    parent_json = "./utils/data/joint24_parents.json"
    edges = get_edges(parent_json)

    enc = Encoder(config, edges)

    enc_info = DotMap({
        'channel_list': enc.channel_list,
        'topologies': enc.topologies,
        'channel_base': enc.channel_base,
        'pooling_list': enc.pooling_list,
        'edge_num': enc.edge_num
    })
    for key, val in enc_info.toDict().items():
        print(f'key={key}, val={val}\n')
    dec = Decoder(config, enc_info)
    enc.cuda()
    dec.cuda()
    enc.eval()
    dec.eval()

    print(enc)
    print(dec)
    train_seq_len = config['train_seq_len']

    with torch.no_grad():
        input = torch.zeros(2, 24*6, train_seq_len).cuda()
        z_vec = enc(input)
        print(z_vec.size())
        out = dec(z_vec)
        print(out.size())


