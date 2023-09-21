# GPU Setting
from config import *
import sys, os

import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Distance, Mask
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import os
import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
import h5py




def CapsNet(input_shape, n_class, routings):
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=8, kernel_size=(5,5), strides=(2,3), padding='same', activation='relu', name='conv1')(x)
    conv2 = layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,2), padding='valid', activation='relu', name='conv2')(conv1)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv2, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='class_caps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Distance(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    return train_model, eval_model

# CapsNet margin loss
def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L, 1))

def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})
                  # metrics=['accuracy', precision, recall, f1score])

    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    model.save_weights(args.save_dir + '/weights.h5')
    print('Trained model saved to \'%s\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model

def normal_dataset():
    x_train_mfcc = []
    y_train_mfcc = []
    x_val_mfcc = []
    y_val_mfcc = []

    for i in range(0, n_label):  # 1~24 label
        for ds_name in ['mfcc', 'y']:
            if ds_name == 'mfcc':
                count = "mfcc_y_%ix%i_%i" % (height, width, i)
                mfcc = h5py.File(save_hdf_train + count + '.h5', 'r')
                x_train_mfcc.extend(mfcc[ds_name])
            if ds_name == 'y':
                count = "mfcc_y_%ix%i_%i" % (height, width, i)
                mfcc = h5py.File(save_hdf_train + count + '.h5', 'r')
                y_train_mfcc.extend(mfcc[ds_name])
                
        for ds_name in ['mfcc', 'y']:
            if ds_name == 'mfcc':
                count = "mfcc_y_%ix%i_%i" % (height, width, i)
                mfcc = h5py.File(save_hdf_validation + count + '.h5', 'r')
                x_val_mfcc.extend(mfcc[ds_name])
            if ds_name == 'y':
                count = "mfcc_y_%ix%i_%i" % (height, width, i)
                mfcc = h5py.File(save_hdf_validation + count + '.h5', 'r')
                y_val_mfcc.extend(mfcc[ds_name])

    train_x, test_x, train_y, test_y = np.array(x_train_mfcc), np.array(x_val_mfcc), np.array(y_train_mfcc), np.array(
        y_val_mfcc)
    train_x = train_x.reshape(-1, height, width, 1)
    test_x = test_x.reshape(-1, height, width, 1)
    train_y = np.argmax(train_y, axis=1).reshape(-1)
    train_y = train_y[:, None]
    test_y = np.argmax(test_y, axis=1).reshape(-1)
    test_y = test_y[:, None]
    train_y = to_categorical(train_y.astype('float32'))
    test_y = to_categorical(test_y.astype('float32'))

    print('normal x:', train_x.shape, test_x.shape)
    print('normal y:', train_y.shape, test_y.shape)

    return (train_x, train_y), (test_x, test_y)

def load_dataset():
    (train_x, train_y), (test_x, test_y) = normal_dataset()
    
    return (train_x, train_y), (test_x, test_y)

def fit():

    import sys;
    sys.argv = [''];
    del sys  # treatment for jupyter's argparse error

    import sys
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    sys.path.append('../')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.Session(config=config)
    set_session(sess)

    K.set_image_data_format('channels_last')

    parser = argparse.ArgumentParser(description="Capsule Network on Dataset.")
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default=model_save)
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load dataset
    (x_train, y_train), (x_test, y_test) = load_dataset()

    # define model
    model, eval_model = CapsNet(input_shape=x_train.shape[1:],
                                n_class=len(np.unique(np.argmax(y_train, 1))),
                                routings=args.routings)
    model.summary()

    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:  # as long as weights are given, will run testing
        print('No weights are provided. Will test using random initialized weights.')

    # save model for testset evaluation
    eval_model.save(args.save_dir + '/eval.h5')

    model_json = model.to_json()  # 모델 아키텍처를 json 형식으로 저장
    save_jfile = args.save_dir + '/model.json'
    with open(save_jfile, "w") as json_file:
        json_file.write(model_json)

if __name__ == "__main__":
    fit()