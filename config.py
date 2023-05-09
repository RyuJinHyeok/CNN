#AAI Machine Learning I _ config.py
device = 'cpu'

SR = 44100
max_len = 4
n_mfcc = 40

# mode
preprocess_mode='MFCC'

# 에포크 설정
num_epochs = 10
# 학습률 설정
LR = 5e-3

# 배치 사이즈 설정
batch_size = 32

# dir_dataset
dir_train = 'C:/Users/ADMIN/Documents/GitHub/data/test/'
dir_validation = 'C:/Users/ADMIN/Documents/GitHub/data/test/'
dir_test = 'C:/Users/ADMIN/Documents/GitHub/data/test/'

# dir_result
model_name = 'ASB_test'
model_save = 'C:/Users/ADMIN/Documents/GitHub/model_4_ASB/%s/' % model_name
result_save = 'C:/Users/ADMIN/Documents/result_4_ASB_TEST/%s/' % model_name

dir_np_save ='.'
