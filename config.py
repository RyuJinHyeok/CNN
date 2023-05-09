
device = 'cpu'

SR = 44100
max_len = 4
n_mfcc = 40

# mode
mode = 'spec'

# 에포크 설정
num_epochs = 100
# 학습률 설정
LR = 5e-3

# 배치 사이즈 설정
batch_size = 32

# dir_dataset
dir_train = 'CapsNet/data/train/'
dir_validation = 'CapsNet/data/validation/'
dir_test = 'CapsNet/data/test/'

# dir_result
model_name = 'specExp2'
model_save = 'CNN/model/%s/' % model_name
result_save = 'CNN/result/%s/' % model_name