from config import *
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras import backend as K
from keras.utils import to_categorical
from capsulelayers import CapsuleLayer, PrimaryCap, Distance, Mask
from keras.backend.tensorflow_backend import set_session
import seaborn as sn
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import sys; sys.argv=['']; del sys
import argparse
import h5py
import time
import csv
import re
import os

# CapsNet margin loss
def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L, 1))

# predict
def model_pred(model, data, args):
    x_test, y_test = data
    start = time.time()
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('\nPrediction response time : ', time.time() - start)
    print('Test accuracy:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

# preprocess data
def id_parse(label_name):
    label = []
    if label_name=='1.차량경적':
        label.append(0)
    elif label_name=='2.차량사이렌':
        label.append(1)
    elif label_name=='3.차량주행음':
        label.append(2)
    elif label_name=='4.이륜차경적':
        label.append(3)
    elif label_name=='5.이륜차주행음':
        label.append(4)
    elif label_name=='6.비행기':
        label.append(5)
    elif label_name=='7.헬리콥터':
        label.append(6)
    elif label_name=='8.기차':
        label.append(7)
    elif label_name=='9.지하철':
        label.append(8)
    elif label_name=='10.발소리':
        label.append(9)
    elif label_name=='11.가구소리':
        label.append(10)
    elif label_name=='12.청소기':
        label.append(11)
    elif label_name=='13.세탁기':
        label.append(12)
    elif label_name=='14.개':
        label.append(13)
    elif label_name=='15.고양이':
        label.append(14)
    elif label_name=='16.공구':
        label.append(15)
    elif label_name=='17.악기':
        label.append(16)
    elif label_name=='18.항타기':
        label.append(17)
    elif label_name=='19.파쇄기':
        label.append(18)
    elif label_name=='20.콘크리트펌프':
        label.append(19)
    elif label_name=='21.발전기':
        label.append(20)
    elif label_name=='22.절삭기':
        label.append(21)
    elif label_name=='23.송풍기':
        label.append(22)
    elif label_name=='24.압축기':
        label.append(23)
    else:
        print('Error : out of index file')
    return label[0]

def test_graph(model, data, args):
    from sklearn.metrics import confusion_matrix
    labels = ['0','1','2','3','4','5']
    ''' 0:총소리 1:응급차량 2:화재/침입경보 3:자동차주행음 4:자동차경적 5:자전거경적 '''
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=20)
    cm = confusion_matrix(np.argmax(y_test, 1), np.argmax(y_pred, 1))
    print('\nCategory Classification Report\n')
    print('0:총소리 1:응급차량 2:화재/침입경보 3:자동차주행음 4:자동차경적 5:자전거경적')
    print(classification_report(np.argmax(y_test, 1), np.argmax(y_pred, 1), target_names=labels))
    print('\nConfusion Matrix')

    print('\nAccuracy')
    y_test_label = np.argmax(y_test, 1)
    y_pred_label = np.argmax(y_pred, 1)
    cnt = np.array([0, 0, 0, 0, 0, 0])

    testset = os.listdir(dir_test)
    for i in range(300):
        if y_test_label[i] == y_pred_label[i]:
            cnt[y_test_label[i]] += 1
        else:
            print(testset[i], y_pred_label[i], y_pred[i])
    
    print(cnt / 50)

    return cm, labels

# read HDF5 file
def test_dataset():
    x_train_mfcc = []
    y_train_mfcc = []

    for i in range(0, n_label):  # 1~6 class
        for ds_name in ['mfcc', 'y']:
            if ds_name == 'mfcc':
                count = "mfcc_y_%ix%i_%i" % (height, width, i)
                mfcc = h5py.File(save_hdf_test + count + '.h5', 'r')
                x_train_mfcc.extend(mfcc[ds_name])
            if ds_name == 'y':
                count = "mfcc_y_%ix%i_%i" % (height, width, i)
                mfcc = h5py.File(save_hdf_test + count + '.h5', 'r')
                y_train_mfcc.extend(mfcc[ds_name])
    # reshape
    test_x = np.array(x_train_mfcc)
    test_y = np.array(y_train_mfcc)
    test_x = test_x.reshape(-1, height, width, 1)
    test_y = np.argmax(test_y, axis=1).reshape(-1)
    test_y = test_y[:, None]
    test_y = to_categorical(test_y.astype('float32'))
    return (test_x, test_y)

def eval():

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    np.random.seed(1337)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.Session(config=config)
    set_session(sess)
    K.set_image_data_format('channels_last')


    parser = argparse.ArgumentParser(description="Capsule Network on Dataset.")
    parser.add_argument('--save_dir', default=result_save)
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load model
    eval_model = load_model('CapsNet/model/base_model/' + 'eval.h5',
                            custom_objects={'CapsuleLayer': CapsuleLayer, 'Mask': Mask, 'Distance': Distance,
                                            'PrimaryCab': PrimaryCap, 'margin_loss': margin_loss})
    eval_model.summary()

    # load test-dataset
    (x_test, y_test) = test_dataset()
    # predict test-dataset
    model_pred(model=eval_model, data=(x_test, y_test), args=args)
    # prediction result
    cm_test, labels = test_graph(model=eval_model, data=(x_test, y_test), args=args)
    # save predict values .csv
    cm_test.tofile(args.save_dir + '/cm_test.csv',',')
    # load predict values .csv
    test = np.loadtxt(open(args.save_dir+'/cm_test.csv', "rb"), delimiter=",", skiprows=0)
    # dataframe
    t = test.reshape(n_label,n_label)
    df_cm = pd.DataFrame(t, labels, labels)
    # heatmap
    plt.figure(figsize = (16,8))
    svn = sn.heatmap(df_cm, annot=True, cmap='Blues', annot_kws={"size": 14}, fmt='g', linewidths=.5)
    figure = svn.get_figure()
    figure.savefig(args.save_dir+'sv_conf.png', dpi=400)

if __name__ == "__main__":
    eval()
