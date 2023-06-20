# -*- coding: utf-8 -*-
import os
import pickle as pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from . import load_stdata
from .preprocessing import preprocess_dataset
from .preprocessing import remove_incomplete_days
from .STMatrix import STMatrix
from .preprocessing import timestamp2vec

import config

np.random.seed(1337)  # for reproducibility


def load_data(T=24, nb_flow=2, len_closeness=None, len_period=1, len_trend=1, len_test=None, data_dir='',
              data_file = '', preprocess_name='preprocessing.pkl', meta_data=True):
    assert(len_closeness > 0)
    # load data
    f_name = os.path.join(config.DATAPATH, data_dir, data_file)
    if not os.path.isfile(f_name):
        preprocess_dataset(config.DATAPATH, data_dir, data_file)

    data, timestamps = load_stdata(f_name)
    data, timestamps = remove_incomplete_days(data, timestamps, T)

    data = data[:, :nb_flow]
    print('data shape: ', data.shape)
    data[data < 0] = 0.
    timestamps_all = [timestamps]

    # normalize
    data_train = data[:-len_test]
    print('train_data shape: ', data_train.shape)
    mmn = MinMaxScaler(feature_range=(-1,1))
    mmn.fit(data_train.reshape(-1,1))
    org_shape = data.shape
    print("min:", mmn.data_min_, "max:", mmn.data_max_)
    data_all_mmn = [mmn.transform(data.reshape(-1,1)).reshape(org_shape)]

    fpkl = open(os.path.join(config.DATAPATH, data_dir, 'preprocessing.pkl'), 'wb')
    for obj in [mmn]:
        pickle.dump(obj, fpkl)
    fpkl.close()

    XC, XP, XT = [], [], []
    Y = []
    timestamps_Y = []
    for data, timestamps in zip(data_all_mmn, timestamps_all):
        # instance-based dataset --> sequences with format as (X, Y) where X is a sequence of images and Y is an image.
        st = STMatrix(data, timestamps, T, CheckComplete=False)
        _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset(len_closeness=len_closeness)
        XC.append(_XC)
        XP.append(_XP)
        XT.append(_XT)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y

    XC = np.vstack(XC)
    XP = np.vstack(XP)
    XT = np.vstack(XT)
    Y = np.vstack(Y)

    print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
    XC_train, XP_train, XT_train, Y_train = XC[:-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
    XC_test, XP_test, XT_test, Y_test = XC[-len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]
    
    timestamp_train, timestamp_test = timestamps_Y[:-len_test], timestamps_Y[-len_test:]
    X_train = []
    X_test = []
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
        if l > 0:
            X_train.append(X_)
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
        if l > 0:
            X_test.append(X_)

    # load meta feature
    if meta_data:
        meta_feature = timestamp2vec(timestamps_Y)
        #print(meta_feature.shape)
        metadata_dim = meta_feature.shape[1]
        meta_feature_train, meta_feature_test = meta_feature[:-len_test], meta_feature[-len_test:]
        X_train.append(meta_feature_train)
        X_test.append(meta_feature_test)
    else:
        metadata_dim = None

    print()
    return X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test
