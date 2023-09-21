from config import *

# modules for data preprocessing (convert wav to mfcc)
from hdf_test import start_hdf_test
from hdf_train import start_hdf_train
from hdf_validation import start_hdf_valid

# module for model trainig
from model_train import fit

# module for model evaluation
from model_eval import eval

# --- NOTICE ---
# you must modify directory paths in config.py

start_hdf_test()
# fit()
eval()