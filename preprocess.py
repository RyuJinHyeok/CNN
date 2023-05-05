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

def preprocess_dataset(data):
    mfccs = []
    for i in data:
        extracted_features = librosa.feature.mfcc(y=i,
                                              sr=SR,
                                              n_mfcc=n_mfcc)
        mfccs.append(extracted_features)
            
    mfccs = np.array(mfccs)

    return mfccs.reshape(-1, 1, mfccs.shape[1], mfccs.shape[2])