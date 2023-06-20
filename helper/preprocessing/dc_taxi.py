import os
import glob
import time
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

    print('Reading csv')
    all_files = glob.glob(os.path.join(data_dir, "*.txt"))
    df_from_each_file = (pd.read_csv(f, sep='|') for f in all_files)
    df = pd.concat(df_from_each_file, ignore_index=True)
    df.dropna(subset=['ORIGIN_BLOCK_LATITUDE', 'ORIGIN_BLOCK_LONGITUDE',
                      'DESTINATION_BLOCK_LATITUDE', 'DESTINATION_BLOCK_LONGITUDE'], inplace=True)
    df['ORIGINDATETIME_TR'] = pd.to_datetime(df['ORIGINDATETIME_TR']).dt.floor('h')
    df['DESTINATIONDATETIME_TR'] = pd.to_datetime(df['DESTINATIONDATETIME_TR']).dt.floor('h')
    df.drop(df[df['ORIGINDATETIME_TR'] < pd.Timestamp(2019, 1, 1)].index, inplace=True)
    df.drop(df[df['ORIGINDATETIME_TR'] > pd.Timestamp(2019, 6, 30)].index, inplace=True)
    df.drop(df[df['DESTINATIONDATETIME_TR'] < pd.Timestamp(2019, 1, 1)].index, inplace=True)
    df.drop(df[df['DESTINATIONDATETIME_TR'] > pd.Timestamp(2019, 6, 30)].index, inplace=True)
    df.sort_values(by='ORIGINDATETIME_TR', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(df.shape)

    return df


def create_dataset(i, df, grids, nx, ny, data):

    point_p = Point(df.loc[i, 'ORIGIN_BLOCK_LONGITUDE'], df.loc[i, 'ORIGIN_BLOCK_LATITUDE'])
    point_d = Point(df.loc[i, 'DESTINATION_BLOCK_LONGITUDE'], df.loc[i, 'DESTINATION_BLOCK_LATITUDE'])
    p_grid_ind = grid_index(grids, point_p)
    d_grid_ind = grid_index(grids, point_d)

    if p_grid_ind != -1:
        r, c = ret_index(p_grid_ind, nx, ny)
        data[df.loc[i, 'ORIGINDATETIME_TR'].floor('h').to_datetime64()][0, r, c] += 1

    if d_grid_ind != -1:
        r, c = ret_index(d_grid_ind, nx, ny)
        data[df.loc[i, 'DESTINATIONDATETIME_TR'].floor('h').to_datetime64()][1, r, c] += 1

    if i % 100000 == 0:
        print(i, sep=' ', end=' ')
        sys.stdout.flush()

def preprocess_dc_taxi(dir, file_name):

    df = load_csv(data_dir=dir)

    nx = 20
    ny = 20
    x_ = [-77.119766, 38.79163, -76.909366, 38.995852]

    grids = divide_map_into_grids(x_, nx, ny)
    data = dict()

    for date_index in pd.unique(df[['ORIGINDATETIME_TR', 'DESTINATIONDATETIME_TR']].values.ravel('K')):
        data[date_index] = torch.zeros(2, ny, nx)

    start = time.time()
    for i in range(len(df)):
        create_dataset(i, df=df, grids=grids, nx=nx, ny=ny, data=data)

    # global out_grid_count
    print((time.time() - start) / 60)
    print('Total Points outside ' + str(out_grid_count) )

    with open(os.path.join(dir, file_name), 'wb') as f:
        pickle.dump(dict(data), f, protocol=4)
