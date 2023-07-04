#패키지 추가
import torch 
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.optim import Adam
from torch.utils.data import DataLoader

#하이퍼 파라미터 설정
batch_size=100
hidden_size=500
num_classes=3
lr=0.001
epochs=3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset=''
test_dataset=''

train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)
