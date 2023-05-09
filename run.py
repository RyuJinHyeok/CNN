# Run.py

'''
수정 로그
# 23.05.04 Run 프로그램 실행 모드 선택 기능 추가
# 23.05.05 dataSet_save 함수 추가 
'''
from dataset import *
from preprocess import *
from train import fit
from eval import *

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

def dataset_save():
    train_wav = train_dataset()
    valid_wav = valid_dataset()

    train_x, train_y = split(train_wav)
    valid_x, valid_y = split(valid_wav)

    train_x = set_length(train_x)
    valid_x = set_length(valid_x)
        
    train_X = preprocess_dataset(train_x)
    valid_X = preprocess_dataset(valid_x)

    np.save("CNN/data/train_X_SAVE",train_X)
    np.save("CNN/data/valid_X_SAVE",valid_X)

    np.save("CNN/data/train_y_SAVE",train_y)
    np.save("CNN/data/valid_y_SAVE",valid_y)

# -------- NOTICE ----------
# you must modify config.py to suit your environment // 실행 시키기 전, config.py를 수정하세요.

seed_everything(929)

''' Run mode Setting'''
# mode 0 : Not working
# mode 1 : Train data processing
# mode 2 : training mode
# mode 3 : start from test data processing + eval


mode = 3

if mode == 1:  # Train data save
    dataset_save()

elif mode == 2: # Train
    train_X_save_load = np.load("CNN/data/spec/train_X_SAVE.npy")
    valid_X_save_load = np.load("CNN/data/spec/valid_X_SAVE.npy")
    train_y_save_load = np.load("CNN/data/spec/train_y_SAVE.npy")
    valid_y_save_load = np.load("CNN/data/spec/valid_y_SAVE.npy")

    fit(train_X_save_load, train_y_save_load, valid_X_save_load, valid_y_save_load, train_X_save_load.shape[3])

elif mode == 3:
    test_wav = test_dataset()
    evaluation_all(test_wav)

