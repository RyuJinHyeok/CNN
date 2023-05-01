from config import *

from tqdm import tqdm
import os
import librosa 
import pandas as pd
import numpy as np

def train_dataset():
    dataset = []
    for file in tqdm(os.listdir(dir_train),colour='green'):
        if 'wav' in file:
            abs_file_path = os.path.join(dir_train, file)
            data, sr = librosa.load(abs_file_path, sr = SR)
            id = os.path.splitext(file)[0]
            class_label = int((-1 if id[0] == 'A' else 2) + int(id[3]))
            dataset.append([data,class_label])
    
    print("Train Dataset 생성 완료")
    return pd.DataFrame(dataset,columns=['data', 'label'])

def valid_dataset():
    dataset = []
    for file in tqdm(os.listdir(dir_validation),colour='green'):
        if 'wav' in file:
            abs_file_path = os.path.join(dir_validation, file)
            data, sr = librosa.load(abs_file_path, sr = SR)
            id = os.path.splitext(file)[0]
            class_label = int((-1 if id[0] == 'A' else 2) + int(id[3]))
            dataset.append([data,class_label])
    
    print("Valid Dataset 생성 완료")
    return pd.DataFrame(dataset,columns=['data','label'])

def test_dataset():
    dataset = []
    for file in tqdm(os.listdir(dir_test),colour='green'):
        if 'wav' in file:
            abs_file_path = os.path.join(dir_test,file)
            data, sr = librosa.load(abs_file_path, sr = SR)
            
            dataset.append([data, file])
    
    print("Test Dataset 생성 완료")
    return pd.DataFrame(dataset,columns=['data', 'file_name'])

def split(df):
    return np.array(df.data), np.array(df.label)