import numpy as np
import pandas as pd
  
from torch.utils import data
from torch.utils.data import Dataset


def get_data(filename):
    data_icu = pd.read_csv(filename).copy()
 

def MyData(data, hrs, label):
    """
    Return x_train, y_train, and other data for the seq2seq model.
    Meanwhile, MinMax each data.
    """
    
    icu_stays_ids = data.iloc[0:, 0].values.copy()
    icu_stays_idx = {}

    for j in np.unique(icu_stays_ids):
        icu_stays_idx[j] = []

    for i in range(len(icu_stays_ids)):
        if i == len(icu_stays_ids) - 1:
            icu_stays_idx[icu_stays_ids[i]].append(i)
            break
        if icu_stays_ids[i] != icu_stays_ids[i + 1]:
            continue
        icu_stays_idx[icu_stays_ids[i]].append(i)

    i_list = []

    for i in icu_stays_idx.keys():
        if len(icu_stays_idx[i]) < hrs * 2:
            i_list.append(i)
        
    for j in i_list:
        icu_stays_idx.pop(j)

    x_train_ = []
    y_train_ = []
    encoder_input = []

    for i in icu_stays_idx.keys():
        icu_stay_label = []
        icu_stay_feature = []
        icu_encoder_input = []
        for j in range(hrs):
            icu_stay_feature.append(data.iloc[icu_stays_idx[i][j], 2:-3].values.copy())
            icu_encoder_input.append(data.iloc[icu_stays_idx[i][j], -3:-2].values.copy())
            icu_stay_label.append(data.iloc[icu_stays_idx[i][j + hrs], label])
        x_train_.append(icu_stay_feature)
        encoder_input.append(icu_encoder_input)
        y_train_.append(icu_stay_label)
    
    return np.array(x_train_), np.array(y_train_), np.array(encoder_input)
  

class myData(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
