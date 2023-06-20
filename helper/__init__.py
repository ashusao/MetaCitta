import pickle
import pandas as pd
import torch

def load_stdata(fname):
    '''f = h5py.File(fname, 'r')
    data = f['data'].value
    timestamps = f['date'].value
    f.close()
    return data, timestamps'''
    # ashu
    f = open(fname, 'rb')
    dict_data = pickle.load(f)
    data = torch.stack(list(dict_data.values())).numpy()
    timestamps = list(dict_data.keys())
    timestamps = [pd.to_datetime(str(t)).strftime('%Y%m%d%H') for t in timestamps]
    f.close()
    return data, timestamps