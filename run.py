from dataset import *
from preprocess import *
from train import fit
from eval import eval

import torch
import pandas as pd
import numpy as np
import os

import random

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



# -------- NOTICE ----------
# you must modify config.py to suit your environment

seed_everything(929)

# load data
train_wav = train_dataset()
valid_wav = valid_dataset()
test_wav = test_dataset()

train_x, train_y = split(train_wav)
valid_x, valid_y = split(valid_wav)


# data preprocessing (zero-padding, mfcc)
train_x = set_length(train_x)
valid_x = set_length(valid_x)

train_X = preprocess_dataset(train_x)
valid_X = preprocess_dataset(valid_x)


# train
fit(train_X, train_y, valid_X, valid_y, train_X.shape[2])

# evaluation
eval(test_wav)