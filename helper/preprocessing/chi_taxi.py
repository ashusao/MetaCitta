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
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    df = pd.concat(df_from_each_file, ignore_index=True)
    df.columns = df.columns.str.replace(' ', '_')
    df.dropna(subset=['Pickup_Centroid_Latitude', 'Pickup_Centroid_Longitude',
                      'Dropoff_Centroid_Latitude', 'Dropoff_Centroid_Longitude'], inplace=True)
    df['Trip_Start_Timestamp'] = pd.to_datetime(df['Trip_Start_Timestamp']).dt.floor('h')
    df['Trip_End_Timestamp'] = pd.to_datetime(df['Trip_End_Timestamp']).dt.floor('h')
    df.drop(df[df['Trip_Start_Timestamp'] < pd.Timestamp(2019, 1, 1)].index, inplace=True)
    df.drop(df[df['Trip_Start_Timestamp'] > pd.Timestamp(2019, 6, 30)].index, inplace=True)
    df.drop(df[df['Trip_End_Timestamp'] < pd.Timestamp(2019, 1, 1)].index, inplace=True)
    df.drop(df[df['Trip_End_Timestamp'] > pd.Timestamp(2019, 6, 30)].index, inplace=True)
    df.sort_values(by='Trip_Start_Timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def create_dataset(i, df, grids, nx, ny, data):

    point_p = Point(df.loc[i, 'Pickup_Centroid_Longitude'], df.loc[i, 'Pickup_Centroid_Latitude'])
    point_d = Point(df.loc[i, 'Dropoff_Centroid_Longitude'], df.loc[i, 'Dropoff_Centroid_Latitude'])

    p_grid_ind = grid_index(grids, point_p)
    d_grid_ind = grid_index(grids, point_d)

    if p_grid_ind != -1:
        r, c = ret_index(p_grid_ind, nx, ny)
        data[df.loc[i, 'Trip_Start_Timestamp'].floor('h').to_datetime64()][0, r, c] += 1

    if d_grid_ind != -1:
        r, c = ret_index(d_grid_ind, nx, ny)
        data[df.loc[i, 'Trip_End_Timestamp'].floor('h').to_datetime64()][1, r, c] += 1

    if i % 100000 == 0:
        print(i, sep=' ', end=' ')
        sys.stdout.flush()

def preprocess_chi_taxi(dir, file_name):
    nx = 20
    ny = 20
    x_ = [-87.858042,41.790506,-87.572397,42.052855]

    grids = divide_map_into_grids(x_, nx, ny)
    df = load_csv(data_dir=dir)

    data = dict()

    for date_index in pd.unique(df[['Trip_Start_Timestamp', 'Trip_End_Timestamp']].values.ravel('K')):
        data[date_index] = torch.zeros(2, ny, nx)

    start = time.time()
    for i in range(len(df)):
        create_dataset(i, df=df, grids=grids, nx=nx, ny=ny, data=data)

    # global out_grid_count
    print((time.time() - start) / 60)
    print('Total Points outside ' + str(out_grid_count) )

    with open(os.path.join(dir, file_name), 'wb') as f:
        pickle.dump(dict(data), f, protocol=4)