'''
Description: 
Author: JinShiyin
Email: shiyinjin@foxmail.com
Date: 2022-04-28 17:57:56
'''
import torch
import torch.nn as nn
from dotmap import DotMap
from lib.models.skeleton import SkeletonUnpool, SkeletonPool, SkeletonConv, find_neighbor, get_edges


class TwoHierSAVAEModel(nn.Module):
    def __init__(self, hp):
        super(TwoHierSAVAEModel, self).__init__()

        self.hp = hp 
        self.latent_d = hp['latent_d']
        self.shallow_latent_d = hp['shallow_latent_d']
        self.n_joints = hp['n_joints']
        self.input_dim = hp['input_dim'] 
        self.output_dim = hp['output_dim']

        parent_json = hp['parent_json_file']
        edges = get_edges(parent_json) # a list with 23(n_joints-1) elements, each elements represents a pair (parent_idx, idx)

        self.enc = Encoder(hp, edges)
        enc_info = DotMap({
            'channel_list': self.enc.channel_list,
            'topologies': self.enc.topologies,
            'channel_base': self.enc.channel_base,
            'pooling_list': self.enc.pooling_list,
            'edge_num': self.enc.edge_num
        })
        self.dec = Decoder(hp, enc_info)
        self.iteration_interval = hp['iteration_interval']

    def forward(self, seq_rot_6d, hp, iterations):
        offset = None 
        bs, timesteps, _ = seq_rot_6d.size()
        seq_rot_6d = seq_rot_6d.transpose(1, 2)
        latent, z_vec_list = self.enc(seq_rot_6d, offset) # input: bs X (n_edges*input_dim) X T
        # latent: bs X (k_edges*d) X (T//2^n_layers)
        # list, each is bs X k_edges X (2*latent_d)
       
        z_list = []
        l_kl_list = []
        for z_idx in range(len(z_vec_list)):
            distributions = z_vec_list[z_idx] # bs X k_edges X (2*latent_d)
            bs, k_edges, _ = distributions.size()
            if z_idx == 0:
                mu = distributions[:, :, :self.shallow_latent_d].view(-1, self.shallow_latent_d) # (bs*k_edges) X latent_d
                logvar = distributions[:, :, self.shallow_latent_d:].view(-1, self.shallow_latent_d) # (bs*k_edges) X latent_d
            else:
                mu = distributions[:, :, :self.latent_d].view(-1, self.latent_d) # (bs*k_edges) X latent_d
                logvar = distributions[:, :, self.latent_d:].view(-1, self.latent_d) # (bs*k_edges) X latent_d

            if hp['kl_w'] != 0:
                z = self.reparametrize(mu, logvar) # (bs*7) X latent_d
            else:
                z = mu # (bs*7) X latent_d

            z = z.view(bs, k_edges, -1) # bs X 7 X latent_d

            iteration_interval = hp['iteration_interval']
   
            if z_idx == len(z_vec_list)-1: # The final deepest layer level
                l_kl_curr = self.kl_loss(logvar, mu) # calculate kl loss with (logvar, mu) and (1, 0) of normal gaussion.
            elif z_idx == 0:
                if iterations < iteration_interval:
                    l_kl_curr = self.kl_loss(logvar.detach(), mu.detach())
                    z = z.detach()
                else:
                    l_kl_curr = self.kl_loss(logvar, mu)
            else:
                l_kl_curr = torch.zeros(1).cuda()

            z_list.append(z)

            l_kl_list.append(l_kl_curr) # From shallow to deep layers

        rec_seq_rot_6d = self.dec(z_list) # bs X (24*6) X T
        rec_seq_rot_6d = rec_seq_rot_6d.transpose(1, 2) # bs X T X (24*6)

        l_kl = hp['kl_w'] * l_kl_list[3] + hp['shallow_kl_w'] * l_kl_list[0]

        return rec_seq_rot_6d, l_kl

    def reparametrize(self, pred_mean, pred_logvar):
        random_z = torch.randn_like(pred_mean)
        vae_z = random_z * torch.exp(0.5 * pred_logvar)
        vae_z = vae_z + pred_mean
        return vae_z

    def kl_loss(self, logvar, mu):
        loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return loss.mean()

    def test(self, seq_rot_6d):
        self.eval()
        with torch.no_grad():
            offset = None 
            bs, timesteps, _ = seq_rot_6d.size() # bs X T X (24*6)
            seq_rot_6d = seq_rot_6d.transpose(1, 2) # bs X (24*6) X T
            latent, z_vec_list = self.enc(seq_rot_6d, offset) # input: bs X (n_edges*input_dim) X T
            # latent: bs X (k_edges*d) X (T//2^n_layers)
            # list, each is bs X k_edges X (2*latent_d)
        
            mean_z_list = []
            sampled_z_list = []
            for z_idx in range(len(z_vec_list)):
                distributions = z_vec_list[z_idx] # bs X k_edges X (2*latent_d)
                bs, k_edges, _ = distributions.size()
                if z_idx == 0:
                    mu = distributions[:, :, :self.shallow_latent_d].view(-1, self.shallow_latent_d) # (bs*k_edges) X latent_d
                    logvar = distributions[:, :, self.shallow_latent_d:].view(-1, self.shallow_latent_d) # (bs*k_edges) X latent_d
                else:
                    mu = distributions[:, :, :self.latent_d].view(-1, self.latent_d) # (bs*k_edges) X latent_d
                    logvar = distributions[:, :, self.latent_d:].view(-1, self.latent_d) # (bs*k_edges) X latent_d

                mean_z = mu # (bs*7) X latent_d
                mean_z = mean_z.view(bs, k_edges, -1) # bs X 7 X latent_d
                mean_z_list.append(mean_z)

                sampled_z = torch.randn_like(mean_z).cuda()
                sampled_z_list.append(sampled_z)

            mean_out_rot_6d = self.dec(mean_z_list) # bs X (24*6) X T
            sampled_out_rot_6d = self.dec(sampled_z_list) # bs X (24*6) X T
            mean_out_rot_6d = mean_out_rot_6d.transpose(1, 2) # bs X T X (24*6)
            sampled_out_rot_6d = sampled_out_rot_6d.transpose(1, 2) # bs X T X (24*6)

        self.train()
        return mean_out_rot_6d, sampled_out_rot_6d


class Encoder(nn.Module):
    def __init__(self, args, topology):
        super(Encoder, self).__init__()
        self.topologies = [topology]
      
        self.latent_d = args['latent_d'] # 24
        self.shallow_latent_d = args['shallow_latent_d'] # 12

        self.channel_base = [6] # 6D representation 
        self.timestep_list = [args['train_seq_len']]
    
        self.channel_list = []
        self.edge_num = [len(topology)]
        self.pooling_list = []
        self.layers = nn.ModuleList()
        self.latent_enc_layers = nn.ModuleList() # Hierarchical latent vectors from different depth of layers 
        self.args = args
        self.convs = []

        kernel_size = args['kernel_size'] # 15
        padding = (kernel_size - 1) // 2
        bias = True

        for i in range(args['num_layers']): # 4
            self.channel_base.append(self.channel_base[-1] * 2) # 6, 12, 24,(48, 96)
          
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
    
            if i == 0:
                latent_encode_linear = nn.Linear(self.channel_base[i+1]*self.timestep_list[i+1], self.shallow_latent_d*2)
            else:
                latent_encode_linear = nn.Linear(self.channel_base[i+1]*self.timestep_list[i+1], self.latent_d*2)
            self.latent_enc_layers.append(latent_encode_linear)

            self.topologies.append(pool.new_edges)
            self.pooling_list.append(pool.pooling_list)
            self.edge_num.append(len(self.topologies[-1]))

    def forward(self, input, offset=None):
        '''
        description: 
        param {*} self
        param {*} input: bs X (24*6) X T
        param {*} offset
        return {*}
        '''
        
        # train_hier_level: 1, 2, 3, 4 (deep to shallow)
        z_vector_list = []
        for i, layer in enumerate(self.layers):
            # print("i:{0}".format(i))
            # print("layer:{0}".format(layer))
            # print("layer input:{0}".format(input.size()))
            input = layer(input)
            # print("layer output:{0}".format(input.size()))
         
            # latent: bs X (k_edges*d) X (T//2^n_layers)
            bs, _, compressed_t = input.size()
            # print("input shape[1]:{0}".format(input.shape[1]))
            # print("channel:{0}".format(self.channel_base[i+1]))
            k_edges = input.shape[1] // self.channel_base[i+1] # edge_num
            # print("k_edges:{0}".format(k_edges))
            
            encoder_map_input = input.view(bs, k_edges, -1)
            # print("encoder_map_input:{0}".format(encoder_map_input.size()))

            curr_z_vector = self.latent_enc_layers[i](encoder_map_input)
            # print("curr_z_vector:{0}".format(curr_z_vector.size()))
            z_vector_list.append(curr_z_vector)
           
        return input, z_vector_list 
        # each z_vector is bs X k_edges X (2*latent_d)


class Decoder(nn.Module):
    def __init__(self, args, enc_info):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.latent_dec_layers = nn.ModuleList() # Hierarchical latent vectors from different depth of layers 

        self.latent_d = args['latent_d']
        self.shallow_latent_d = args['shallow_latent_d']

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
            if i == args['num_layers'] - 1:
                in_channels = self.enc.channel_list[args['num_layers'] - i]*2
            else:
                in_channels = self.enc.channel_list[args['num_layers'] - i] # 7*96, 9*48, 14*24, 24*12
            # print("in channels:{0}".format(in_channels))
            if i == args['num_layers'] - 1:
                out_channels = in_channels // 4
            else:   
                out_channels = in_channels // 2 # 7*96->7*24, 9*48->9*12, 14*24->14*6, 24*12->24*6
            
            neighbor_list = find_neighbor(self.enc.topologies[args['num_layers'] - i - 1], args['skeleton_dist']) # 7, 9, 14, 24
            # neighbor_list = find_neighbor(enc.topologies[args['num_layers'] - i], args['skeleton_dist']) # 7, 7, 9, 14

            if i != 0 and i != args['num_layers'] - 1:
                bias = False
            else:
                bias = True
          
            if i == args['num_layers'] - 1:
                latent_decode_linear = nn.Linear(self.shallow_latent_d, self.enc.channel_base[args['num_layers']-i]*self.timestep_list[i]) # 96*4
            else:
                latent_decode_linear = nn.Linear(self.latent_d, self.enc.channel_base[args['num_layers']-i]*self.timestep_list[i]) # 24*8, 12*16, 6*32
            self.latent_dec_layers.append(latent_decode_linear)

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
            self.convs.append(seq[-1])
            if i != args['num_layers'] - 1: seq.append(nn.LeakyReLU(negative_slope=0.2))

            self.layers.append(nn.Sequential(*seq))

    def forward(self, z_vec_list, offset=None):
        # train_hier_level: 1, 2, 3, 4 (deep to shallow)
        hier_feats = []
        num_z_vec = len(z_vec_list)
        for z_idx in range(len(z_vec_list)):
            curr_z_vector = z_vec_list[num_z_vec - z_idx - 1] # bs X k_edges X latent_d
            # print("curr_z_vec:{0}".format(curr_z_vector.size()))
            curr_feats = self.latent_dec_layers[z_idx](curr_z_vector) # each is bs X k_edges X (d*timesteps)
            # print("before view curr feats:{0}".format(curr_feats.size()))
            bs = curr_z_vector.size()[0]
            curr_feats = curr_feats.view(bs, -1, self.timestep_list[z_idx]) # bs X (k_edges*d_feats) X T'
            # print("curr_feats:{0}".format(curr_feats.size()))
            hier_feats.append(curr_feats) # from deep to shallow layer feats

        # hier feats: bs X (7*96) X 4, bs X (7*24) X 8, bs X (9*12) X 16, bs X (14*6) X 32
        for i, layer in enumerate(self.layers): # From deep to shallow layers
            # print("Decoder i:{0}".format(i))
            # print("Decoder layer:{0}".format(layer))
            if i == 0:
                input = hier_feats[i] # bs X (k_edges*d) X T'
            elif i == self.hp['num_layers'] - 1:
                bs, k_d, compressed_t = input.size()
                k_edges = self.enc.edge_num[self.hp['num_layers']-i]
                # print("decoder forward k edges:{0}".format(k_edges))
                tmp_input = input.view(bs, k_edges, -1, compressed_t)
                tmp_hier_feats = hier_feats[i].view(bs, k_edges, -1, compressed_t)
                
                tmp_cat_feats = torch.cat((tmp_input, tmp_hier_feats), dim=2) # bs X k_edges X (d+d') X T'
                input = tmp_cat_feats.view(bs, -1, compressed_t) # bs X (k_edges*2d) X T'
               
            # print("Decoder layer input:{0}".format(input.size()))
            input = layer(input)
            # print("Decoder layer output:{0}".format(input.size()))
            
        return input

