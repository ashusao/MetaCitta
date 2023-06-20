input_len = 12

map_height, map_width = 20, 20  # grid size
nb_flow = 2

epoch_nums = 500
update_lr = 1e-5
meta_lr = 1e-5
batch_size = 64
params = {'batch_size': batch_size,
          'shuffle': False,
          'drop_last': True,
          'num_workers': 0
          }

validation_split = 0.1
early_stop_patience = 30
shuffle_dataset = True

T = 24
days_test = 30
len_test = T * days_test

epoch_save = [0, epoch_nums - 1] \
             + list(range(0, epoch_nums, 50))  # 1*1000

# adjust the path to data directory
DATAPATH = '/path/to/data/'

source_city_data_dir = ['CHIBike', 'DCBike', 'NYCBike', 'DCTaxi', 'NYCTaxi']
source_city_dataset_name = ['bikechi', 'bikedc', 'bikenyc', 'taxidc', 'taxinyc']

# index mapping for task of same city
spatial_map = [(1, 3), (2, 4)]
# index mapping of same task across the cities
domain_map = [(1, 2), (3, 4)]

alphas = [1.0]

data_file = 'demand.pkl'
model_name = 'conv_lstm'

meta_dir = 'Taxi_CHI'