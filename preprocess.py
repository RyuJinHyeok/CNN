from config import *

import numpy as np
import librosa
from tqdm import tqdm

def set_length(data):

    print('set length')

    length = max_len * SR

    result = []
    for i in tqdm(data):

        if len(i) > length:
            arr = i[:length]
        else:
            arr = np.pad(i, (0, length - len(i)), mode='constant', constant_values=0)
        
        result.append(arr.tolist())
        
    result = np.array(result)

    return result

def preprocess_dataset_MFCC(data):
    mfccs = []
    for i in data:
        extracted_features = librosa.feature.mfcc(y=i,
                                              sr=SR,
                                              n_mfcc=n_mfcc)
        mfccs.append(extracted_features)
            
    mfccs = np.array(mfccs)

    return mfccs.reshape(-1, 1, mfccs.shape[1], mfccs.shape[2])

def preprocess_dataset_melSpec(data):
    specs = []
    for i in tqdm(data):
        extracted_features = librosa.feature.melspectrogram(y=i,
                                              sr=SR)
        specs.append(extracted_features)
            
    specs = np.array(specs)

    return specs.reshape(-1, 1, specs.shape[1], specs.shape[2])

def preprocess_dataset_stft(data):
    stfts = []
    for i in tqdm(data):
        extracted_features = librosa.stft(y=i, n_fft=512, hop_length=512)
        stfts.append(extracted_features)
            
    stfts = np.array(stfts)

    return stfts.reshape(-1, 1, stfts.shape[1], stfts.shape[2])