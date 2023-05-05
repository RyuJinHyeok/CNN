from config import *
from dataLoader import *
from cnnlayers import *

import torch
import torch.optim as optim # 최적화 알고리즘들이 포함됨
from torch.utils.data import DataLoader # 학습 및 배치로 모델에 넣어주기 위한 툴


from tqdm.auto import tqdm
import os

import matplotlib.pyplot as plt
import numpy as np

def fit(train_X, train_y, valid_X, valid_y, mfcc_y):
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #GPU 할당

    train_data = CustomDataset(X=train_X, y=train_y)
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True)

    valid_data = CustomDataset(X=valid_X, y=valid_y)
    valid_loader = DataLoader(valid_data, batch_size = batch_size, shuffle=False)


    model = CNNclassification().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(params = model.parameters(), lr = 1e-3 )
    scheduler = None

    model(torch.rand(batch_size, 1, n_mfcc, mfcc_y).to(device))

    train(model, criterion, optimizer, train_loader, valid_loader, scheduler, device)


def train(model, criterion, optimizer, train_loader, valid_loader, scheduler, device): 

    print('\n------------ training start ---------------')

    train_loss = []
    val_loss = []

    train_acc = []
    val_acc = []

    model.to(device)
    n = len(train_loader)
    best_acc = 0
    
    for epoch in range(1,num_epochs + 1): #에포크 설정
        model.train() #모델 학습
        running_loss = 0.0
        
        correct = 0
        for wav, label in tqdm(iter(train_loader)):
            
            wav, label = wav.to(device), label.to(device) #배치 데이터
            optimizer.zero_grad() #배치마다 optimizer 초기화
        
            # Data -> Model -> Output
            logit = model(wav) #예측값 산출
            loss = criterion(logit, label) #손실함수 계산
            pred = logit.argmax(dim=1, keepdim=True)  #10개의 class중 가장 값이 높은 것을 예측 label로 추출
            correct += pred.eq(label.view_as(pred)).sum().item() #예측값과 실제값이 맞으면 1 아니면 0으로 합산
            
            # 역전파
            loss.backward() #손실함수 기준 역전파 
            optimizer.step() #가중치 최적화
            running_loss += loss.item()
             
        print('[%d] Train loss: %.10f' %(epoch, running_loss / len(train_loader)))

        train_loss.append(running_loss / len(train_loader))
        train_acc.append(correct / len(train_loader.dataset))

        if scheduler is not None:
            scheduler.step()
            
            
        #Validation set 평가
        model.eval() #evaluation 과정에서 사용하지 않아야 하는 layer들을 알아서 off 시키도록 하는 함수
        vali_loss = 0.0
        correct = 0
       
        with torch.no_grad(): #파라미터 업데이트 안하기 때문에 no_grad 사용
            for wav, label in tqdm(iter(valid_loader)):
                
                wav, label = wav.to(device), label.to(device)
                logit = model(wav)
                vali_loss += criterion(logit, label)
                pred = logit.argmax(dim=1, keepdim=True)  #10개의 class중 가장 값이 높은 것을 예측 label로 추출
                correct += pred.eq(label.view_as(pred)).sum().item() #예측값과 실제값이 맞으면 1 아니면 0으로 합산
        vali_acc = 100 * correct / len(valid_loader.dataset)
        print('Vali set: Loss: {:.4f}, Accuracy: {}/{} ( {:.0f}%)\n'.format(vali_loss / len(valid_loader), correct, len(valid_loader.dataset), 100 * correct / len(valid_loader.dataset)))
        
        val_loss.append(vali_loss / len(valid_loader))
        val_acc.append(correct / len(valid_loader.dataset))

        #베스트 모델 저장
        if best_acc < vali_acc:
            best_acc = vali_acc

            if not os.path.exists(model_save):
                os.makedirs(model_save)

            torch.save(model.state_dict(), model_save + model_name + '.pth') #이 디렉토리에 best_model.pth을 저장
            print('Model Saved.')
        

    def draw_result(train, val, mode=0):
        x = np.arange(1, num_epochs + 1, 1)

        plt.plot(x, train, label='train')
        plt.plot(x, val, label='validation')

        plt.title('accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy' if mode == 0 else 'loss')

        plt.legend()
        
        if mode == 0:
            plt.savefig(model_save + 'accuarcy.png')
        else:
            plt.savefig(model_save + 'loss.png')

    draw_result(train_acc, val_acc)
    plt.clf()
    draw_result(train_loss, val_loss, 1)


