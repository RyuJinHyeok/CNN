# eval.py
'''
수정 로그
# 23.05.04 Test 결과 저장 기능들 추가
'''

from config import *
from preprocess import *
from dataLoader import *
from cnnlayers import *

from torch.utils.data import DataLoader # 학습 및 배치로 모델에 넣어주기 위한 툴

import torch
from tqdm import tqdm
import numpy as np

# 사이킷런 #
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt
import json

import pandas as pd
# 예측 수행
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


# 결과 출력
def result(df,preds,expInfo=None): # 결과 생성 
    # 답안지, 정답지 생성
    df['answer'] = 0
    df['predict'] = preds

    #파일 이름에서 라벨 추출
    df.loc[df['file_name'].str.contains('A_01_'), 'answer'] = 0
    df.loc[df['file_name'].str.contains('A_02_'), 'answer'] = 1
    df.loc[df['file_name'].str.contains('A_03_'), 'answer'] = 2
    df.loc[df['file_name'].str.contains('B_01_'), 'answer'] = 3
    df.loc[df['file_name'].str.contains('B_02_'), 'answer'] = 4
    df.loc[df['file_name'].str.contains('B_03_'), 'answer'] = 5
    df['answer'] = df['answer'].astype(int)  # 정답(aswer) colum을 정수형으로 변환
    
    answer_arr = df['answer'].tolist()
    # anslist_arr=np.array(anslist)  # 정답 넘파이 배열
    # preds_arr=np.array(preds) # 예측 넘파이 배열
    

    ''' 평가용 결과 출력'''

    """1. Confusion_matrix 출력"""
    cm = confusion_matrix(answer_arr,preds) # 정답(x축)값, 예측(Y축)값을 이용하여 Confusion Matrix 계산
    plt.title('Confusion Matrix') # 제목 추가
    sns.heatmap(cm, annot=True, cmap='Blues') # 도시
    plt.savefig('%sconfusion_matrix_%s.png'%(result_save, expInfo)) # 그래프를 이미지 파일로 저장


    """2. 상세 로그 저장 """
    df.drop('data',axis=1,inplace=True)  # 소리 데이터 제거
    df.to_csv("%sTest_evaluation_all_%s.csv" % (result_save, expInfo), index=False) # data frame 

    
    for i in range(len(preds)):
        if df['answer'][i]==df['predict'][i]:
            df=df.drop(i, axis=0)
    # rs=df.drop_duplicates(subset=['answer', 'predict'], inplace=False) # 오분류 사례만 추출
    df.to_csv("%sTest_evaluation_error_case_%s.csv"%(result_save, expInfo), index=False) # data frame 


    """3. 보고서 생성 및 저장"""
    # 예측값과 실제값을 이용하여 정확도 계산
    acc = accuracy_score(answer_arr,preds)
    acc_report = "%s 모델의 Test 정확도 (합산) : "%model_name+'{:.4f}'.format(acc)
    print(acc_report)  #결과 출력

    with open("%s/acuracy_report.txt" %result_save , "w") as f:
        f.write(acc_report)

    # 분류 보고서 생성
    report = classification_report(answer_arr,preds)
    print(report)
 
    # 결과를 json으로 저장
    with open("%s/classification_report_%s.json" %(result_save,expInfo), "w") as outfile:
        json.dump(report, outfile)

    


    



#모델 평가
def eval(test_wav):     # Test 데이터 전처리 포함됨.
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
    df = test_wav.copy()

    # 답안지, 정답지 생성
    df['answer'] = 0
    df['label'] = preds
    #파일 이름에서 라벨 추출
    df.loc[df['file_name'].str.contains('A_01_'), 'answer'] = 0
    df.loc[df['file_name'].str.contains('A_02_'), 'answer'] = 1
    df.loc[df['file_name'].str.contains('A_03_'), 'answer'] = 2
    df.loc[df['file_name'].str.contains('B_01_'), 'answer'] = 3
    df.loc[df['file_name'].str.contains('B_02_'), 'answer'] = 4
    df.loc[df['file_name'].str.contains('B_03_'), 'answer'] = 5
    df['answer'] = df['answer'].astype(int)  # 정답 라벨을 정수형으로 변환

    anslist = df['answer'].tolist()
    anslist_arr=np.array(anslist)  # 정답 넘파이 배열
    preds_arr=np.array(preds) # 예측 넘파이 배열
    

    # Confusion_matrix 도시
    cm = confusion_matrix(anslist_arr,preds_arr)     # 예측값과 실제값을 이용하여 혼동행렬 계산
    # 히트맵 그리기
    sns.heatmap(cm, annot=True, cmap='Blues')

# 제목 추가
    plt.title('Confusion Matrix')

# 그래프를 이미지 파일로 저장
    plt.savefig('heatmap.jpg')
    # plt.savefig('confusion_matrix.png')

    # 예측값과 실제값을 이용하여 정확도 계산
    acc = accuracy_score(anslist_arr,preds_arr)
    print('모델 전체 정확도: {:.4f}'.format(acc))  # 결과 출력


    # 분류 보고서 생성
    report = classification_report(anslist_arr,preds_arr)
    print(report)

    # 자료 저장
    df.drop('data',axis=1,inplace=True)  # 소리 데이터 제거
    df.to_csv("Test_detail.csv", index=False) # data frame 저장

    
def evaluation_all(test_wav):
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
    df = test_wav.copy()
    del test_wav
    result(df,preds,expInfo='1')










'''
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
'''