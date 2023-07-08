import torch
import torch.nn as nn
class myLeNet(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
        #배치 노말라이즈 = 값이 큰 애들, 작은 애들끼리 각각 노말라이징해서 분포를 적당하게 맞춤.
        #몇 개의 채널을 넣을 건지 명시해줘야.
        self.bn1 =nn.BatchNorm2d(num_features=6)
        self.act1=nn.ReLU()
        self.pool1=nn.MaxPool2d(kernel_size=2)
        
        self.conv2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        
        self.bn2 =nn.BatchNorm2d(num_features=16)
        self.act2=nn.ReLU()
        self.pool2=nn.MaxPool2d(kernel_size=2)
        
        self.fc1=nn.Linear(in_features=16*5*5,out_features=120)
        self.fc2=nn.Linear(in_features=120,out_features=84)
        self.fc3=nn.Linear(in_features=84,out_features=num_classes)
       
    #forward에서는 데이터를 하나 받게 돼있음. 그게 x. 일종의 약속임.
    #만약 x가 튜플이면 x=(a,b)같은 식으로 강제로 나눠줌.
    def forward(self, x):
        b,c,h,w = x.shape
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.act1(x)
        x=self.pool1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.act2(x)
        x=self.pool2(x)
        
        x=x.reshape(b,-1)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        return x