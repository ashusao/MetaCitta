# -*- coding: utf-8 -*-
import sys
import time

sys.path.append('.')
import os
import torch.nn as nn
import torch
from torch.utils import data

from helper.make_dataset import make_dataloader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from model import ConvLSTM
from utils import compute_errors
from torch_two_sample import MMDStatistic

import config


out_dir = '/data/sao/pretraining/tmp/' + config.meta_dir + '/reports'
checkpoint_dir = out_dir + '/checkpoint'
writer_dir = out_dir + '/runs/' + config.model_name + '/'

os.makedirs(checkpoint_dir + '/%s' % (config.model_name), exist_ok=True)
writer = SummaryWriter(writer_dir)

random_seed = 1337


def valid(model, val_generators, criterion, device):
    model.eval()
    mean_loss = []

    for val_generator in val_generators:
        for i, (X_c, _, _, _, Y_batch) in enumerate(val_generator):
            # Move tensors to the configured device
            X_c = X_c.type(torch.FloatTensor).to(device)
            outputs, _, _, _, _, _ = model(X_c)
            mse, _, _ = criterion(outputs.cpu().data.numpy(), Y_batch.data.numpy())

            mean_loss.append(mse)

    mean_loss = np.mean(mean_loss)
    #print('Mean valid loss:', mean_loss)

    return mean_loss

def create_generators(dataset):
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(config.validation_split * dataset_size))
    if config.shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_gen = data.DataLoader(dataset, **config.params, sampler=train_sampler)
    val_gen = data.DataLoader(dataset, **config.params, sampler=valid_sampler)

    iters = int(np.ceil(len(train_indices) / config.batch_size) * config.epoch_nums)
    batches = int(np.ceil(len(train_indices) / config.batch_size))

    return train_gen, val_gen, iters, batches

def fwd_pass(train_iter, train_generator, model, device, loss_fn):

    try:
        X_c, X_p, X_t, X_meta, Y_batch = next(train_iter)
    except StopIteration:
        # print('train exhausted ...')
        train_iter = iter(train_generator)  # create iter once it is exhausted
        X_c, _, _, _, Y_batch = next(train_iter)  # extract the data

    # Move tensors to the configured device
    X_c = X_c.type(torch.FloatTensor).to(device)
    Y_batch = Y_batch.type(torch.FloatTensor).to(device)

    # Update
    outputs, conv_out, _, _, fc_out, fc1_out = model(X_c)

    loss = loss_fn(outputs, Y_batch)

    return loss, fc_out, conv_out, fc1_out


def update_mmd(model, opt, device, train_iters, train_generators, loss_fn, loss_fn_mmd):

    cum_loss = 0.0
    mmd_loss = 0.0

    for m_ in config.spatial_map:

        opt.zero_grad()

        loss_t, fc_t, conv_t, fc1_t = fwd_pass(train_iters[m_[0]], train_generators[m_[0]], model, device, loss_fn)
        loss_s, fc_s, conv_s, fc1_s = fwd_pass(train_iters[m_[1]], train_generators[m_[1]], model, device, loss_fn)

        loss_mmd = loss_fn_mmd(fc1_t.reshape(fc1_t.size(0), -1), fc1_s.reshape(fc1_s.size(0), -1), config.alphas)

        cum_loss += (loss_s.item() + loss_t.item())
        mmd_loss += loss_mmd.item()

        (loss_s + loss_t + loss_mmd).backward()

        opt.step()


    for m_ in config.domain_map:

        opt.zero_grad()

        loss_t, fc_t, conv_t, fc1_t = fwd_pass(train_iters[m_[0]], train_generators[m_[0]], model, device, loss_fn)
        loss_s, fc_s, conv_s, fc1_s = fwd_pass(train_iters[m_[1]], train_generators[m_[1]], model, device, loss_fn)

        loss_mmd = loss_fn_mmd(fc_t.reshape(fc_t.size(0), -1), fc_s.reshape(fc_s.size(0), -1), config.alphas)

        cum_loss += (loss_s.item() + loss_t.item())
        mmd_loss += loss_mmd.item()

        (loss_s + loss_t + loss_mmd).backward()

        opt.step()

    return cum_loss / (2.0*(len(train_iters)-1)) , mmd_loss / (len(train_iters)-1)


def update(model, opt, device, train_iter, train_generator, loss_fn):

    loss, _, _, _ = fwd_pass(train_iter, train_generator, model, device, loss_fn)

    opt.zero_grad()
    loss.backward()
    opt.step()

    return loss.item()

def train():

    source_city_train_datasets = []
    source_city_train_generators = []
    source_city_val_generators = []

    for i, dataset in enumerate(config.source_city_dataset_name):
        source_city_train_datasets.append(make_dataloader(dataset_name=dataset, mode='train',
                                                          len_closeness=config.input_len,
                                                          data_dir=config.source_city_data_dir[i],
                                                          data_file=config.data_file))

    total_iters = 0
    num_batches = 0

    for train_dataset in source_city_train_datasets:
        train_generator, val_generator, iters, batches = create_generators(train_dataset)

        source_city_train_generators.append(train_generator)
        source_city_val_generators.append(val_generator)

        # Total iterations
        total_iters = max(total_iters, iters)
        num_batches = max(num_batches, batches)

    # create iterators
    source_city_train_iters = [iter(generator) for generator in source_city_train_generators]

    model = ConvLSTM(conf=(config.input_len, config.nb_flow, config.map_height, config.map_width))
    X_c, _, _, _, _ = next(iter(source_city_train_generators[0]))
    writer.add_graph(model, [X_c.type(torch.FloatTensor)])

    # Loss and optimizer
    loss_fn = nn.MSELoss()
    loss_fn_mmd = MMDStatistic(config.batch_size, config.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.meta_lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn.to(device)

    with torch.autograd.set_detect_anomaly(True):
        for e in range(config.epoch_nums):
            b = 0
            model.train()
            # until all the batches are over
            while b < num_batches:

                # mmd update
                loss_s, mmd_loss_train = update_mmd(model, optimizer, device, source_city_train_iters,
                                           source_city_train_generators,
                                           loss_fn, loss_fn_mmd)

                # outer optimization
                loss_t = update(model, optimizer, device, source_city_train_iters[0],
                                           source_city_train_generators[0],
                                           loss_fn)

                b += 1

            its = b + e * num_batches

            val_loss = valid(model, source_city_val_generators, compute_errors, device)

            writer.add_scalars('loss', {'train': (loss_t + loss_s) / 2.0,
                                        'val': val_loss}, its)

            print('Epoch [{}/{}], step [{}/{}], train Loss: {:.4f}, val Loss: {:.4f}, mmd Loss: {:.4f}'
                  .format(e + 1, config.epoch_nums, its, total_iters, (loss_t + loss_s) / 2.0,
                          val_loss, mmd_loss_train))

            if e in config.epoch_save:
                torch.save(model.state_dict(), checkpoint_dir + '/%s/%08d_model.pth' % (config.model_name, e))
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'iter': its,
                    'epoch': e,
                }, checkpoint_dir + '/%s/%08d_optimizer.pth' % (config.model_name, e))


if __name__ == '__main__':

    start = time.time()
    train()
    stop = time.time()

    with open('elapsed_time.txt', 'a+') as f:
        print('Training Time for ' + config.meta_dir + ' : ' + str((stop - start) / 3600.0) + 'hours', file=f)
