from config import *
from preprocess import *
from dataLoader import *
from cnnlayers import *

from torch.utils.data import DataLoader # 학습 및 배치로 모델에 넣어주기 위한 툴

import torch
from tqdm import tqdm
import numpy as np

def predict(model, test_loader, device):
    
    model.eval()
    model_pred = []
    with torch.no_grad():
        for wav in tqdm(iter(test_loader)):
            wav = wav.to(device)

            pred_logit = model(wav)
            pred_logit = pred_logit.argmax(dim=1, keepdim=True).squeeze(1)

            model_pred.extend(pred_logit.tolist())
    return model_pred

def eval(test_wav):

    test_x = np.array(test_wav.data)
    test_x = set_length(test_x)
    test_X = preprocess_dataset(test_x)

    test_data = CustomDataset(X=test_X, y= None, train_mode=False)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle=False)


    checkpoint = torch.load(model_save + model_name + '.pth')
    model = CNNclassification().to(device)
    model.load_state_dict(checkpoint)

    # Inference
    preds = predict(model, test_loader, device)
    
    test_wav['label'] = preds
    test_wav = test_wav[['file_name', 'label']]

    pred_df = test_wav.copy()
    pred_df = pred_df.sort_values(by=[pred_df.columns[0]], ascending=[True]).reset_index(drop=True)

    correct = 0
    corr_list = [0, 0, 0, 0, 0, 0]
    cnt = [0, 0, 0, 0, 0, 0]
    for index in range(0, len(pred_df)):
        data = pred_df.loc[index]
        answer = (-1 if data.file_name[0] == 'A' else 2) + int(data.file_name[3])
        cnt[answer] += 1
        if answer == data.label:
            corr_list[answer] += 1
            correct += 1

    print(corr_list)

    for i in range(6):
        print('acc:', corr_list[i] / cnt[i])

    accuracy = correct / len(pred_df)
    print('correct : %d / %d' % (correct, len(pred_df)))
    print('test accuracy : %.4f' % accuracy)