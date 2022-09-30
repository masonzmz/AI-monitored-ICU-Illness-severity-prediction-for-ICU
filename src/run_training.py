from numpy import newaxis

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score

import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from datetime import datetime
from statsmodels.tsa import stattools
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from load_data import *

def main():
    # LSTM 48hrs dropout: 0.1 seed: 4 learning rate: 4 mse_testing: 2.55
    # GRU  48hrs dropout: 0.1 seed: 1 learning rate: 4 mse_testing: 2.78
    # RNN  48hrs dropout: 0.1 seed: 4 learning rate: 4 mse_testing: 2.91
    input_seqs = []

    duration = 24

    num_features = 24
    batch_size = 128
    hidden_layer_size = 128
    
    # Number of hidden layers
    hidden_layers = 3

    # Weight decay for given optimiser
    decay = 8
    
    # Training iterations (CHANGE)
    num_epochs = 100

    learning_rate_list = [1, 2, 3, 4, 5]
    seeds = [1, 2, 3, 4, 5]
    nn_list = ['gru', 'rnn', 'lstm']

    # optimizer = torch.optim.SGD()
    # optimizer = torch.optim.Adagrad
    # optimizer = torch.optim.AdamW
    # optimizer = torch.optim.Adadelta
    
    X, Y, EncoderInput = MyData(data_icu, 24, -3)
    scalers = MinMaxScaler(feature_range=(0, 1))

    X_train = X.copy()
    Y_train = Y.copy()

    scalers_ = [MinMaxScaler(feature_range=(0, 1)) for i in range(X_train.shape[1])]
    for i in range(X_train.shape[1]):
        X_train[:, i, :] = scalers_[i].fit_transform(X_train[:, i, :])
        
    mse_train = torch.nn.MSELoss()

    mse_test = torch.nn.MSELoss()
    mae_test = torch.nn.L1Loss()
    mape_test = MapeLoss()
    rmse_test = RMSELoss()
    rs_test = RSLoss()
    
    optimizer = torch.optim.AdamW
    
    rounds = len(learning_rate_list) * len(seeds) * len(nn_list)
    round_num = 0

    for nn_name in nn_list:
        for lr in learning_rate_list:
            if nn_name == 'gru':
                dropout = 0.1
            else:
                dropout = 0.2
            for seed in seeds:
                round_num = round_num + 1
                print(f'{round_num}/{rounds} rounds...')
                print(f'Dropout: {dropout}, Learning rate: {lr}')
                train_icu_mse(lr, decay, dropout, seed, nn_name, duration)


if __name__ == '__main__':
    main()
