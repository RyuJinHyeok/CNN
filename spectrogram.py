from dataset import *
from preprocess import *
from tqdm import tqdm

def preprocess_dataset(data):
    specs = []
    for i in tqdm(data):
        extracted_features = librosa.feature.melspectrogram(y=i,
                                              sr=SR)
        specs.append(extracted_features)
            
    specs = np.array(specs)

    return specs.reshape(-1, 1, specs.shape[1], specs.shape[2])


def dataset_save():
    train_wav = train_dataset()
    valid_wav = valid_dataset()

    train_x, train_y = split(train_wav)
    valid_x, valid_y = split(valid_wav)

    train_x = set_length(train_x)
    valid_x = set_length(valid_x)
        
    train_X = preprocess_dataset(train_x)
    valid_X = preprocess_dataset(valid_x)

    np.save("CNN/data/spec/train_X_SAVE",train_X)
    np.save("CNN/data/spec/valid_X_SAVE",valid_X)

    np.save("CNN/data/spec/train_y_SAVE",train_y)
    np.save("CNN/data/spec/valid_y_SAVE",valid_y)

# dataset_save()

# train_X_save_load = np.load("CNN/data/spec/train_X_SAVE.npy")
# print(train_X_save_load)

# np.save("CNN/data/spec/train_X_SAVE1.npy", train_X_save_load[:480])
# np.save("CNN/data/spec/train_X_SAVE2.npy", train_X_save_load[480:960])
# np.save("CNN/data/spec/train_X_SAVE3.npy", train_X_save_load[960:1440])
# np.save("CNN/data/spec/train_X_SAVE4.npy", train_X_save_load[1440:1920])
# np.save("CNN/data/spec/train_X_SAVE5.npy", train_X_save_load[1920:])