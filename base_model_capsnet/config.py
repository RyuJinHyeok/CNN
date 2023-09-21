import glob

# label category number 0~23
n_label = 6

# MFCC param
SR = 44100 # [Hz] sampling rate
max_len = 4.0
max_len = int(max_len)
n_fft = 2048
n_hop = 1024
n_mfcc = 48
len_raw = int(SR * max_len)

# MFCC image size
height = 48 # Input image height
width = 173 # Input image width

# dir_result
model_name = 'test1'
model_save = 'CapsNet/model/%s/' % model_name
result_save = 'CapsNet/result/%s/' % model_name

# dir_dataset
dir_train = 'CapsNet/data/train/'
dir_validation = 'CapsNet/data/validation/'
dir_test = 'CapsNet/data/test/'

# dif_hdf
save_hdf_train = 'CapsNet/hdf_data/hdf_train/'
save_hdf_validation = 'CapsNet/hdf_data/hdf_validation/'
save_hdf_test = 'CapsNet/hdf_data/hdf_test/'

# folder pattern
pattern_train = dir_train + '*.wav'
file_path_train = glob.glob(pattern_train)

pattern_validation = dir_validation + '*.wav'
file_path_validation = glob.glob(pattern_validation)

pattern_test = dir_test + '*.wav'
file_path_test = glob.glob(pattern_test)