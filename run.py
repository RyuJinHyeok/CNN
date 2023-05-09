#AAI Machine Learning I _ Run.py
'''
modification log / 수정 이력
# 23.05.04 Run 프로그램 실행 모드 선택 기능 추가
# 23.05.05 dataSet_save 함수 추가 
'''

'''library import'''
# Python Library
import torch
import numpy as np
import os
import random

#Custom Library 
from dataset import *  #데이터셋 생성
from preprocess import *  #전처리
from train import fit  # 모델 학습
from eval import * #평가

'''Random Seed'''
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# ------------------
'''0. numpy save Function'''
# -- 매개변수 안내-- #
# dataset Type - 'train' : 학습용 , 'valid' : 검증용, 'test' : 평가용 , 파일 이름 자동 지정
# preprocess - 전처리 유형 - MFCC: MFCC , Spec: spectrogram 
# savePath - 저장 경로
def np_Dataset_save(dir_np_save,dataSetTpye, preprocess, data_X,data_y):
    np.save("%s/%s_%s_X_SAVE" %(dir_np_save, preprocess,dataSetTpye),data_X)
    np.save("%s/%s_%s_y_SAVE" %(dir_np_save, preprocess,dataSetTpye), data_y)
        
'''1. Make MFCC dataSet'''
def Train_dataset_save_MFCC(): # 모델 Train MFCC 데이터 세트 생성
    print("MFCC data set")
    # Wav 파일 : col1 = wav signal, col2 = 정답 lable (파일 이름에서 추출) --> Pandas dataFrame 반환
    train_wav = train_dataset()  # Train data
    valid_wav = valid_dataset()  # Validation data

    # wav signal과 정답 label 분리
    train_x, train_y = split(train_wav) # Train data
    valid_x, valid_y = split(valid_wav) # Validation data

    # padding 수행
    train_x = set_length(train_x) # Train data
    valid_x = set_length(valid_x) # Validation data

    # MFCC로 데이터 전처리 수행
    train_X = preprocess_dataset_MFCC(train_x) # Train data
    valid_X = preprocess_dataset_MFCC(valid_x) # Validation data

    # numpy 배열로 데이터 Set 저장
    np_Dataset_save(dir_np_save,'train', preprocess_mode, train_X,train_y)
    np_Dataset_save(dir_np_save,'valid', preprocess_mode, valid_X,valid_y)

'''2. Make Mel-spectorgram dataSet'''
def Train_dataset_save_melSpec(): # 모델 Train MFCC 데이터 세트 생성
    print("Mel-spectrogram data set")
    # Wav 파일 : col1 = wav signal, col2 = 정답 lable (파일 이름에서 추출) --> Pandas dataFrame 반환
    train_wav = train_dataset()  # Train data
    valid_wav = valid_dataset()  # Validation data

    # wav signal과 정답 label 분리
    train_x, train_y = split(train_wav) # Train data
    valid_x, valid_y = split(valid_wav) # Validation data

    # padding 수행
    train_x = set_length(train_x) # Train data
    valid_x = set_length(valid_x) # Validation data

    # mel-spectroram로 데이터 전처리 수행
    train_X = preprocess_dataset_melSpec(train_x) # Train data
    valid_X = preprocess_dataset_melSpec(valid_x) # Validation data

    # numpy 배열로 데이터 Set 저장
    np_Dataset_save(dir_np_save,'train', preprocess_mode, train_X,train_y)
    np_Dataset_save(dir_np_save,'valid', preprocess_mode, valid_X,valid_y)
# -------- NOTICE ----------
# you must modify config.py to suit your environment // 실행 시키기 전, config.py를 수정하세요.

seed_everything(929)  # Random Seed

''' Run mode Setting'''
# mode 0 : Not working
# mode 1 : Train data processing
# mode 2 : training mode
# mode 3 : start from test data processing + eval


mode = 2

if mode == 1:  # Train data save
    if preprocess_mode == 'MFCC':
        print("Train data save=MFCC")
        Train_dataset_save_MFCC()

    elif preprocess_mode == 'spec':
        print("Train data save=spectrogram")
        Train_dataset_save_melSpec()

elif mode == 2: # Train
    train_X_save_load = np.load("%s/%s_train_X_SAVE.npy"%(dir_np_save,preprocess_mode))
    train_y_save_load = np.load("%s/%s_train_y_SAVE.npy"%(dir_np_save,preprocess_mode))
    valid_X_save_load = np.load("%s/%s_valid_X_SAVE.npy"%(dir_np_save,preprocess_mode))
    valid_y_save_load = np.load("%s/%s_valid_y_SAVE.npy"%(dir_np_save,preprocess_mode))
    fit(train_X_save_load, train_y_save_load, valid_X_save_load, valid_y_save_load, train_X_save_load.shape[3])

elif mode == 3:
    test_wav = test_dataset()
    evaluation_all(test_wav)

