'''
Description: 
Author: JinShiyin
Email: shiyinjin@foxmail.com
Date: 2022-04-30 17:38:59
'''
from tkinter.messagebox import NO
import torch
import joblib
import os.path as osp
from torch.utils.data import Dataset
from lib.utils.dataset_utils import split_into_chunks

class AMASS(Dataset):
    def __init__(self, seqlen, overlap, db_dir, mode='train', logger=None):
        self.seqlen = seqlen
        self.stride = int((1-overlap)*seqlen)
        self.db_dir = db_dir
        self.mode = mode

        self.db = self.load_db()
        self.vid_indices = split_into_chunks(self.db['vid_name'], self.seqlen, self.stride)
        del self.db['vid_name']
        if logger is not None:
            logger.info(f'AMASS {mode} dataset number of videos: {len(self.vid_indices)}')
        else:
            print(f'AMASS {mode} dataset number of videos: {len(self.vid_indices)}')

    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, index):
        return self.get_single_item(index)

    def load_db(self):
        db_path = osp.join(self.db_dir, f'amass_{self.mode}_db_fps30.pt')
        db = joblib.load(db_path)
        return db

    def get_single_item(self, index):
        start_index, end_index = self.vid_indices[index]
        thetas = self.db['theta'][start_index:end_index+1]
        thetas = torch.from_numpy(thetas).float()
        return thetas # T X (24*3+24*3)= T X 144


def get_all_loaders(cfg, logger):

    workers = cfg['num_workers']

    train_dataset = AMASS(seqlen=cfg['train_seq_len'], overlap=cfg['overlap'], db_dir=cfg['db_dir'], mode='train', logger=logger)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg['train_batch_size'], shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=True)

    val_dataset = AMASS(seqlen=cfg['train_seq_len'], overlap=cfg['overlap'], db_dir=cfg['db_dir'], mode='val', logger=logger)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg['val_batch_size'], shuffle=False,
        num_workers=1, pin_memory=True, drop_last=True)

    test_dataset = AMASS(seqlen=cfg['train_seq_len'], overlap=cfg['overlap'], db_dir=cfg['db_dir'], mode='test', logger=logger)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg['test_batch_size'], shuffle=True,
        num_workers=1, pin_memory=True, drop_last=True)

    return (train_loader, val_loader, test_loader)