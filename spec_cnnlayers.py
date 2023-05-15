from tqdm.auto import tqdm
import torch
import torch.nn as nn # 신경망들이 포함됨

class SpecCNNclassification(torch.nn.Module):
    def __init__(self):
        super(SpecCNNclassification, self).__init__()
        self.layer1 = torch.nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 7), stride=1, padding=0), #cnn layer
            nn.ReLU(), #activation function
            nn.MaxPool2d(kernel_size=(3, 4), stride=(3, 4), padding=0)) #pooling layer
        
        self.layer2 = torch.nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 5), stride=1, padding=0), #cnn layer
            nn.ReLU(), #activation function
            nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 3), padding=0)) #pooling layer
        
        self.layer3 = torch.nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 4), stride=1, padding=0), #cnn layer
            nn.ReLU(), #activation function
            nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 3), padding=0)) #pooling layer
        
        self.layer4 = torch.nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=(0, 0)), #cnn layer
            nn.ReLU(), #activation function
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)) #pooling layer
        
        self.dropout = nn.Dropout()
        
        self.relu = nn.ReLU()

        self.fc_layer1 = nn.Sequential( 
            nn.Linear(768, 768), #fully connected layer(ouput layer)
        )

        self.fc_layer2 = nn.Sequential( 
            nn.Linear(768, 256), #fully connected layer(ouput layer)
        )    

        self.fc_layer3 = nn.Sequential( 
            nn.Linear(256, 6), #fully connected layer(ouput layer)
        )    
        
    def forward(self, x):
        
        x = self.layer1(x.float()) #1층

        x = self.layer2(x) #2층

        x = self.dropout(x)

        x = self.layer3(x) #3층

        x = self.dropout(x)

        x = self.layer4(x) #4층

        x = self.dropout(x)

        x = torch.flatten(x, start_dim=1) # N차원 배열 -> 1차원 배열

        x = self.relu(self.fc_layer1(x))

        x = self.dropout(x)

        x = self.relu(self.fc_layer2(x))

        x = self.dropout(x)

        out = self.fc_layer3(x)
        return out
    
model = SpecCNNclassification().to('cuda')
import torchsummary as t
t.summary(model, (1, 128, 345))