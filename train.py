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
