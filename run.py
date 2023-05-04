# Run.py

'''
수정 로그
# 23.05.04 Run 프로그램 실행 모드 선택 기능 추가
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



# -------- NOTICE ----------
# you must modify config.py to suit your environment // 실행 시키기 전, config.py를 수정하세요.

seed_everything(929)

''' Run mode Setting'''
Train = False # 학습 모드
Test = True # 평가모드
#Test_set_gen = True # True : 테스트용 학습 데이터를 생성한 후 평가 시행 , False : 


if Train == True:
    # load data
    train_wav = train_dataset()
    valid_wav = valid_dataset()

    train_x, train_y = split(train_wav)
    valid_x, valid_y = split(valid_wav)

    train_x = set_length(train_x)
    valid_x = set_length(valid_x)
    
    # data preprocessing (zero-padding, mfcc)
    train_X = preprocess_dataset(train_x)
    valid_X = preprocess_dataset(valid_x)
 
    # train
    fit(train_X, train_y, valid_X, valid_y, train_X.shape[2])

if Test ==True:
    test_wav = test_dataset()
    # evaluation
    evaluation_all(test_wav)