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
        for month in range(7, 13):
            urllib.request.urlretrieve(
                "https://divvy-tripdata.s3.amazonaws.com/2020{0:0=2d}-divvy-tripdata.zip".format(month),
                data_dir + "/2020{0:0=2d}-divvy-tripdata.zip".format(month))
            with zipfile.ZipFile( data_dir + "/2020{0:0=2d}-divvy-tripdata.zip".format(month), "r") as zip_ref:
                zip_ref.extractall(data_dir)

    print('Reading csv')
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    df = pd.concat(df_from_each_file, ignore_index=True)
    df['started_at'] = pd.to_datetime(df['started_at']).dt.floor('h')
    df['ended_at'] = pd.to_datetime(df['ended_at']).dt.floor('h')
    df.dropna(subset=['end_lat', 'end_lng'], inplace=True)
    df.drop(df[df['started_at'] < pd.Timestamp(2020, 7, 1)].index, inplace=True)
    df.drop(df[df['started_at'] > pd.Timestamp(2020, 12, 31)].index, inplace=True)
    df.drop(df[df['ended_at'] < pd.Timestamp(2020, 7, 1)].index, inplace=True)
    df.drop(df[df['ended_at'] > pd.Timestamp(2020, 12, 31)].index, inplace=True)
    df.sort_values(by='started_at', inplace=True)
    df = df.reset_index(drop=True)
    print(df.shape)

    return df


def create_dataset(i, df, grids, nx, ny, data):

    point_p = Point(df.loc[i, 'start_lng'], df.loc[i, 'start_lat'])
    point_d = Point(df.loc[i, 'end_lng'], df.loc[i, 'end_lat'])
    p_grid_ind = grid_index(grids, point_p)
    d_grid_ind = grid_index(grids, point_d)

    if p_grid_ind != -1:
        r, c = ret_index(p_grid_ind, nx, ny)
        data[df.loc[i, 'started_at'].floor('h').to_datetime64()][0, r, c] += 1

    if d_grid_ind != -1:
        r, c = ret_index(d_grid_ind, nx, ny)
        data[df.loc[i, 'ended_at'].floor('h').to_datetime64()][1, r, c] += 1

    if i % 100000 == 0:
        print(i, sep=' ', end=' ')
        sys.stdout.flush()


def preprocess_chi_bike(dir, file_name):
    nx = 20
    ny = 20
    x_ = [-87.858042, 41.790506, -87.572397, 42.052855]

    grids = divide_map_into_grids(x_, nx, ny)
    df = load_csv(data_dir=dir)

    data = dict()

    for date_index in pd.unique(df[['started_at', 'ended_at']].values.ravel('K')):
        data[date_index] = torch.zeros(2, ny, nx)

    start = time.time()
    for i in range(len(df)):
        create_dataset(i, df=df, grids=grids, nx=nx, ny=ny, data=data)

    # global out_grid_count
    print((time.time() - start) / 60)
    print('Total Points outside ' + str(out_grid_count) )

    with open(os.path.join(dir, file_name), 'wb') as f:
        pickle.dump(dict(data), f, protocol=4)