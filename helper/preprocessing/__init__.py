import numpy as np
import time
import os
from .nyc_taxi import preprocess_nyc_taxi
from .nyc_bike import preprocess_nyc_bike
from .chi_bike import preprocess_chi_bike
from .dc_bike import preprocess_dc_bike
from .dc_taxi import preprocess_dc_taxi
from .chi_taxi import preprocess_chi_taxi

def preprocess_dataset(datapath, dataset, datafile):
    if dataset == 'NYCTaxi':
        preprocess_nyc_taxi(os.path.join(datapath, dataset), datafile)
    if dataset == 'NYCBike':
        preprocess_nyc_bike(os.path.join(datapath, dataset), datafile)
    if dataset == 'CHIBike':
        preprocess_chi_bike(os.path.join(datapath, dataset),  datafile)
    if dataset == 'DCBike':
        preprocess_dc_bike(os.path.join(datapath, dataset),  datafile)
    if dataset == 'DCTaxi':
        preprocess_dc_taxi(os.path.join(datapath, dataset),  datafile)
    if dataset == 'CHITaxi':
        preprocess_chi_taxi(os.path.join(datapath, dataset),  datafile)

def timestamp2vec(timestamps):
    # tm_wday range [0, 6], Monday is 0
    vec = [time.strptime(str(t[:8]), '%Y%m%d').tm_wday for t in timestamps]  # python3  # ashu encoding = utf-8 removed
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        ret.append(v)
    return np.asarray(ret)


def remove_incomplete_days(data, timestamps, T=48):
    # remove a certain day which has not 48 timestamps
    days = []  # available days: some day only contain some seqs
    days_incomplete = []
    i = 0
    while i < len(timestamps):
        if int(timestamps[i][8:]) != 0:  #ashu 1 --> 0
            i += 1
        elif i+T-1 < len(timestamps) and int(timestamps[i+T-1][8:]) == T-1:  #ashu T --> T-1
            days.append(timestamps[i][:8])
            i += T
        else:
            days_incomplete.append(timestamps[i][:8])
            i += 1
    print("incomplete days: ", days_incomplete)
    days = set(days)
    idx = []
    for i, t in enumerate(timestamps):
        if t[:8] in days:
            idx.append(i)

    data = data[idx]
    timestamps = [timestamps[i] for i in idx]
    return data, timestamps






