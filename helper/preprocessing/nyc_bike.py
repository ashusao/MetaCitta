import os
import glob
import time
import urllib
import zipfile
import pickle
import sys

import pandas as pd
from shapely.geometry import Point
import torch

from .grid_processor import divide_map_into_grids
from .grid_processor import ret_index
from .grid_processor import grid_index
from .grid_processor import out_grid_count


def load_csv(data_dir):
    if not glob.glob(os.path.join(data_dir, '*.csv')):
        print('Downloading data')
        for month in range(1, 7):
            urllib.request.urlretrieve("https://s3.amazonaws.com/tripdata/" + \
                                       "2019{0:0=2d}-citibike-tripdata.csv.zip".format(month), data_dir + \
                                       "/2019{0:0=2d}-citibike-tripdata.csv.zip".format(month))

            with zipfile.ZipFile(data_dir + "/2019{0:0=2d}-citibike-tripdata.csv.zip".format(month), "r") as zip_ref:
                zip_ref.extractall(data_dir)

    print('Reading csv')
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    df = pd.concat(df_from_each_file, ignore_index=True)
    df.starttime = pd.to_datetime(df.starttime).dt.floor('h')
    df.stoptime = pd.to_datetime(df.stoptime).dt.floor('h')
    df.drop(df[df['starttime'] < pd.Timestamp(2019, 1, 1)].index, inplace=True)
    df.drop(df[df['starttime'] > pd.Timestamp(2019, 6, 30)].index, inplace=True)
    df.drop(df[df['stoptime'] < pd.Timestamp(2019, 1, 1)].index, inplace=True)
    df.drop(df[df['stoptime'] > pd.Timestamp(2019, 6, 30)].index, inplace=True)
    # df = pd.read_csv(os.path.join(data_dir, 'nyc.2019-01.csv'))
    df.sort_values(by='starttime', inplace=True)
    df = df.reset_index(drop=True)
    print(df.shape)

    return df


def create_dataset(i, df, grids, nx, ny, data):

    point_p = Point(df.loc[i, 'start station longitude'], df.loc[i, 'start station latitude'])
    point_d = Point(df.loc[i, 'end station longitude'], df.loc[i, 'end station latitude'])
    p_grid_ind = grid_index(grids, point_p)
    d_grid_ind = grid_index(grids, point_d)

    if p_grid_ind != -1:
        r, c = ret_index(p_grid_ind, nx, ny)
        data[df.loc[i, 'starttime'].floor('h').to_datetime64()][0, r, c] += 1

    if d_grid_ind != -1:
        r, c = ret_index(d_grid_ind, nx, ny)
        data[df.loc[i, 'stoptime'].floor('h').to_datetime64()][1, r, c] += 1

    if i % 100000 == 0:
        print(i, sep=' ', end=' ')
        sys.stdout.flush()

def preprocess_nyc_bike(dir, file_name):
    nx = 20
    ny = 20
    x_ = [-74.197292,40.525562,-73.638383,40.940952]

    grids = divide_map_into_grids(x_, nx, ny)
    df = load_csv(data_dir=dir)

    data = dict()
    for date_index in pd.unique(df[['starttime', 'stoptime']].values.ravel('K')):
        data[date_index] = torch.zeros(2, ny, nx)

    start = time.time()
    for i in range(len(df)):
        create_dataset(i, df=df, grids=grids, nx=nx, ny=ny, data=data)

    # global out_grid_count
    print((time.time() - start) / 60)
    print('Total Points outside ' + str(out_grid_count))

    with open(os.path.join(dir, file_name), 'wb') as f:
        pickle.dump(dict(data), f, protocol=4)
