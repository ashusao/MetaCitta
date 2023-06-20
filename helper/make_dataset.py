# -*- coding: utf-8 -*-
from torch.utils import data
import sys
sys.path.append('.')

from . import Dataloader
import config

class make_dataloader(data.Dataset):
    def __init__(self, dataset_name, mode, len_closeness, data_dir, data_file):
        'Initialization'
        self.dataset_name = dataset_name
        self.mode = mode
        self.len_closeness = len_closeness
        self.data_dir = data_dir

        if self.dataset_name == 'taxinyc' or self.dataset_name == 'bikenyc' or self.dataset_name == 'bikesf' or \
                self.dataset_name == 'bikechi' or self.dataset_name == 'bikedc' or self.dataset_name == 'taxidc' or \
                self.dataset_name == 'taxichi':
            print("loading data...")

            if self.mode == 'train':
                self.X_data, self.Y_data, _, _, mmn, external_dim, self.timestamp_Y, _ = Dataloader.load_data(
                    len_closeness=self.len_closeness,
                    len_test=config.len_test,
                    data_dir=data_dir,
                    data_file=data_file,
                    preprocess_name='preprocessing.pkl',
                    meta_data=True)


            elif self.mode == 'test':
                _, _, self.X_data, self.Y_data, mmn, external_dim, _, self.timestamp_Y = Dataloader.load_data(
                    len_closeness=self.len_closeness,
                    len_test=config.len_test,
                    data_dir=data_dir,
                    data_file=data_file,
                    preprocess_name='preprocessing.pkl',
                    meta_data=True)

            assert len(self.X_data[0]) == len(self.Y_data)
            self.data_len = len(self.Y_data)

        else:
            print('Unknown datasets')

        self.mmn = mmn

    def __len__(self):
        'Denotes the total number of samples'
        return self.data_len

    def __str__(self):
        string = '' \
                 + '\tmode   = %s\n' % self.mode \
                 + '\tdataset name   = %s\n' % self.dataset_name \
                 + '\tmmn min   = %d\n' % self.mmn._min \
                 + '\tmmn max   = %d\n' % self.mmn._max \
                 + '\tlen    = %d\n' % len(self)

        return string

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X_c, X_p, X_t, X_meta = self.X_data[0][index], self.X_data[1][index], self.X_data[2][index], self.X_data[3][index]
        y = self.Y_data[index]
        ts_Y = self.timestamp_Y[index]

        return X_c, X_p, X_t, X_meta, y

    def denormalize(self, d):
        org_shape = d.shape
        return self.mmn.inverse_transform(d.reshape(-1, 1)).reshape(org_shape)
