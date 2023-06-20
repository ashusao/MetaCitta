import os
import glob
import sys
import time

import urllib.request
import zipfile
import pandas as pd
import torch
import pickle

from shapely.geometry import Point, shape
from shapely.ops import transform
import pyproj
import fiona

from .grid_processor import divide_map_into_grids
from .grid_processor import ret_index
from .grid_processor import grid_index
from .grid_processor import out_grid_count

def get_lat_lon(taxi_zone):
    content = []
    # transform to 4326
    wgs84 = pyproj.CRS('EPSG:4326')
    init_crs = taxi_zone.crs
    project = pyproj.Transformer.from_crs(init_crs, wgs84, always_xy=True).transform

    for zone in taxi_zone:
        bbox = shape(zone['geometry']).bounds
        loc_id = zone['properties']['OBJECTID']

        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2

        # compute center point
        bbox_c = transform(project, Point(x, y))

        content.append((loc_id, bbox_c.x, bbox_c.y))
    return pd.DataFrame(content, columns=["OBJECTID", "longitude", "latitude"])

def load_csv(data_dir):

    if not glob.glob(os.path.join(data_dir, '*.csv')):
        print('Downloading data')
        for month in range(1, 7): # for testing only
            urllib.request.urlretrieve("https://s3.amazonaws.com/nyc-tlc/trip+data/" + \
                                       "yellow_tripdata_2019-{0:0=2d}.csv".format(month),
                                       data_dir + "/nyc.2019-{0:0=2d}.csv".format(month))
        urllib.request.urlretrieve("https://s3.amazonaws.com/nyc-tlc/misc/taxi_zones.zip",
                                   data_dir + "/taxi_zones.zip")
        with zipfile.ZipFile(data_dir + "/taxi_zones.zip", "r") as zip_ref:
            zip_ref.extractall(data_dir + "/shape")
    print('Reading csv')
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    df = pd.concat(df_from_each_file, ignore_index=True)
    #df = pd.read_csv(os.path.join(data_dir, 'nyc.2019-01.csv'))
    #df = df.head(200000)
    print(df.shape)

    # filter rows
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.floor('h')
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime']).dt.floor('h')
    df.drop(df[df['tpep_pickup_datetime'] < pd.Timestamp(2019, 1, 1)].index, inplace=True)
    df.drop(df[df['tpep_pickup_datetime'] > pd.Timestamp(2019, 6, 30)].index, inplace=True)
    df.drop(df[df['tpep_dropoff_datetime'] < pd.Timestamp(2019, 1, 1)].index, inplace=True)
    df.drop(df[df['tpep_dropoff_datetime'] > pd.Timestamp(2019, 6, 30)].index, inplace=True)
    df.drop(df[df['PULocationID'] >= 264].index, inplace=True)  # drop unknown zones
    df.drop(df[df['DOLocationID'] >= 264].index, inplace=True)  # drop unknown zones
    print(df.shape)

    # transform zone to lat lon
    taxi_zone = fiona.open(os.path.join(data_dir,'shape','taxi_zones.shp'))
    # extract properties from shape file
    fields_name = list(taxi_zone.schema['properties'])
    # create dictionary of properties and its values
    shp_attr = [dict(zip(fields_name, zone['properties'].values())) for zone in taxi_zone]
    # create dataframe
    df_loc = pd.DataFrame(shp_attr).join(get_lat_lon(taxi_zone).set_index("OBJECTID"), on="OBJECTID")
    # df_loc = pd.DataFrame(shp_attr)
    df_loc.drop_duplicates(inplace=True)
    print(df_loc.shape)

    locid_long_dict = pd.Series(df_loc.longitude.values, index=df_loc.OBJECTID).to_dict()
    locid_lat_dict = pd.Series(df_loc.latitude.values, index=df_loc.OBJECTID).to_dict()

    df['PU_latitude'] = df['PULocationID'].map(locid_lat_dict)
    df['PU_longitude'] = df['PULocationID'].map(locid_long_dict)
    df['DO_latitude'] = df['DOLocationID'].map(locid_lat_dict)
    df['DO_longitude'] = df['DOLocationID'].map(locid_long_dict)

    df.sort_values(by='tpep_pickup_datetime', inplace=True)
    df = df.reset_index(drop=True)
    print(df.head())

    return df


def create_dataset(i, df, grids, nx, ny, data):

    point_p = Point(df.loc[i, 'PU_longitude'], df.loc[i, 'PU_latitude'])
    point_d = Point(df.loc[i, 'DO_longitude'], df.loc[i, 'DO_latitude'])

    p_grid_ind = grid_index(grids, point_p)
    d_grid_ind = grid_index(grids, point_d)

    if p_grid_ind != -1:
        r, c = ret_index(p_grid_ind, nx, ny)
        data[df.loc[i, 'tpep_pickup_datetime'].floor('h').to_datetime64()][0,r,c] += 1

    if d_grid_ind != -1:
        r, c = ret_index(d_grid_ind, nx, ny)
        data[df.loc[i, 'tpep_dropoff_datetime'].floor('h').to_datetime64()][1,r,c] += 1

    if i % 100000 == 0:
        print(i, sep=' ', end=' ')
        sys.stdout.flush()

def preprocess_nyc_taxi(dir, file_name):
    nx = 20
    ny = 20
    x_ = [-74.197292, 40.525562, -73.638383, 40.940952]

    grids = divide_map_into_grids(x_, nx, ny)
    df = load_csv(data_dir=dir)

    data = dict()

    for date_index in pd.unique(df[['tpep_pickup_datetime', 'tpep_dropoff_datetime']].values.ravel('K')):
        data[date_index] = torch.zeros(2, ny, nx)

    start = time.time()
    for i in range(len(df)):
        create_dataset(i, df=df, grids=grids, nx=nx, ny=ny, data=data)

    #global out_grid_count
    print((time.time() - start) / 60)
    print('Total Points outside ' + str(out_grid_count))

    with open(os.path.join(dir, model_dir, file_name), 'wb') as f:
        pickle.dump(dict(data), f, protocol=4)

