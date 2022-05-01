'''
Description: 
Author: JinShiyin
Email: shiyinjin@foxmail.com
Date: 2022-05-01 13:03:57
'''
import torch.nn as nn
from fk_layer import ForwardKinematicsLayer
from skeleton import SkeletonPool, SkeletonConv, find_neighbor, get_edges


class TrajectoryModel(nn.Module):
    def __init__(self, hp):
        super(TrajectoryModel, self).__init__()

        self.latent_d = hp['latent_d']
        self.n_joints = hp['n_joints']
        self.input_dim = hp['input_dim'] 
        self.output_dim = hp['output_dim']
        self.max_timesteps = hp['train_seq_len']

        parent_json = "./utils/data/joint24_parents.json"
        edges = get_edges(parent_json) # a list with 23(n_joints-1) elements, each elements represents a pair (parent_idx, idx)

        self.fk_layer = ForwardKinematicsLayer()

        self.hp = hp 

        self.enc = Encoder(hp, edges)
         
        self.d_model = self.enc.channel_base[-1]
      
        # self.fc_mapping = nn.Linear(self.d_model, 3)
        self.fc_mapping = nn.Linear(self.d_model*7, 3)
    

    def forward(self, input):
        '''
        description: 
        param {*} self
        param {*} input: bs X T X (24*n_dim), n_dim can be 3 or 3+2
        return {*}
        '''
        offset = None 
        bs, timesteps, _ = input.size()
        input = input.transpose(1, 2) # bs X (24*6) X T
        latent = self.enc(input, offset) # bs X (n_edges*d) X T
        k_edges = latent.shape[1] // self.d_model # the edge_num of latent
        encoder_map_input = latent.view(bs, k_edges, self.d_model, timesteps) # bs X k_edges X d X T
        encoder_map_input = encoder_map_input.transpose(2, 3).transpose(1, 2) # bs X T X k_edges X d
        pred_root_v = self.fc_mapping(encoder_map_input.view(bs, timesteps, -1)) # bs X T X 3

        return pred_root_v


class Encoder(nn.Module):
    def __init__(self, args, topology):
        super(Encoder, self).__init__()
        self.topologies = [topology]
        # if args.rotation == 'euler_angle': self.channel_base = [3]
        # elif args.rotation == 'quaternion': self.channel_base = [4]
        if args['trajectory_input_joint_pos']:
            self.channel_base = [3] # 6 + 3
        else:
            self.channel_base = [6] 
        self.channel_list = []
        self.edge_num = [len(topology)]
        self.pooling_list = []
        self.layers = nn.ModuleList()
        self.args = args
        self.convs = []

        kernel_size = args['kernel_size']
        padding = (kernel_size - 1) // 2
        bias = True

        for i in range(args['num_layers']):
            self.channel_base.append(self.channel_base[-1] * 2) # 6, 12, 24, 48, 96

        for i in range(args['num_layers']):
            seq = []
            neighbor_list = find_neighbor(self.topologies[i], args['skeleton_dist'])
            in_channels = self.channel_base[i] * self.edge_num[i]
            out_channels = self.channel_base[i+1] * self.edge_num[i]
            # print("edge num:{0}".format(self.edge_num[i]))
            # print("in channels:{0}".format(in_channels))
            # print("out channels:{0}".format(out_channels))
            if i == 0: self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)

            for _ in range(args['extra_conv']):
                seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=in_channels,
                                        joint_num=self.edge_num[i], kernel_size=kernel_size, stride=1,
                                        padding=padding, padding_mode=args['padding_mode'], bias=bias))
            # print("i:{0}".format(i))
            # print("neighbor list:{0}".format(neighbor_list))
            
            curr_stride = 1
        
            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=self.edge_num[i], kernel_size=kernel_size, stride=curr_stride,
                                    padding=padding, padding_mode=args['padding_mode'], bias=bias, add_offset=False,
                                    in_offset_channel=3 * self.channel_base[i] // self.channel_base[0]))
            self.convs.append(seq[-1])
            last_pool = True if i == args['num_layers'] - 1 else False
            pool = SkeletonPool(edges=self.topologies[i], pooling_mode=args['skeleton_pool'],
                                channels_per_edge=out_channels // len(neighbor_list), last_pool=last_pool)
            seq.append(pool)
            seq.append(nn.LeakyReLU(negative_slope=0.2))
            self.layers.append(nn.Sequential(*seq))

            # self.topologies.append(topology)
            self.topologies.append(pool.new_edges)
            self.pooling_list.append(pool.pooling_list)
            # print("topology:{0}".format(self.topologies))
            # print("pool list:{0}".format(self.pooling_list))
            self.edge_num.append(len(self.topologies[-1]))

    def forward(self, input, offset=None):
        for i, layer in enumerate(self.layers):
            # print("layer input:{0}".format(input.size()))
            input = layer(input)
            # print("layer output:{0}".format(input.size()))
            # import pdb 
            # pdb.set_trace()
        return input
