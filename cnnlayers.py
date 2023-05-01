from tqdm.auto import tqdm
import torch
import torch.nn as nn # 신경망들이 포함됨

class CNNclassification(torch.nn.Module):
    def __init__(self):
        super(CNNclassification, self).__init__()
        self.layer1 = torch.nn.Sequential(
            nn.Conv2d(40, 32, kernel_size=2, stride=1, padding=1), #cnn layer
            nn.ReLU(), #activation function
            nn.MaxPool2d(kernel_size=4, stride=4, padding=1)) #pooling layer
        
        self.layer2 = torch.nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1), #cnn layer
            nn.ReLU(), #activation function
            nn.MaxPool2d(kernel_size=4, stride=4, padding=1)) #pooling layer
        
        self.layer3 = torch.nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1), #cnn layer
            nn.ReLU(), #activation function
            nn.MaxPool2d(kernel_size=4, stride=4, padding=1)) #pooling layer
        
        self.layer4 = torch.nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=1), #cnn layer
            nn.ReLU(), #activation function
            nn.MaxPool2d(kernel_size=4, stride=4, padding=1)) #pooling layer
        
        self.dropout = nn.Dropout(p=0.2)

        self.fc_layer = nn.Sequential( 
            nn.Linear(512, 6) #fully connected layer(ouput layer)
        )    
        
    def forward(self, x):
        
        x = self.layer1(x.float()) #1층
        x = self.dropout(x)
        # print(x.shape)
        x = self.layer2(x) #2층
        x = self.dropout(x)
        # print(x.shape)
        x = self.layer3(x) #3층
        x = self.dropout(x)
        # print(x.shape)
        x = self.layer4(x) #4층
        x = self.dropout(x)
        # print(x.shape)
        # x = self.dropout(x)
        x = torch.flatten(x, start_dim=1) # N차원 배열 -> 1차원 배열
        
        out = self.fc_layer(x)
        return out