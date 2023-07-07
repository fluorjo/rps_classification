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

#mlp 클래스 정의
class myMLP(nn.Module):
    #객체 생성. linear 4개.
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.fc1=nn.Linear(28*28,hidden_size)
        self.fc2=nn.Linear(hidden_size,hidden_size)
        self.fc3=nn.Linear(hidden_size,hidden_size)
        self.fc4=nn.Linear(hidden_size,hidden_size)
    #데이터 크기 변환
    def forward(self, x):
        b,w,h,c = x.shape
        x=x.reshape(-1,28*28)
    #레이어에 태우기.
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        x=self.fc4(x)
    # 출력.
        return x

#mlp 클래스의 모델을 device로 전송 및 실행시킴. 
model=myMLP(hidden_size,num_classes).to(device)
#로스 함수. crossentropy
loss=nn.CrossEntropyLoss()
#옵티마이저.
optim=Adam(model.parameters(),lr=lr)

#학습시키기.
#epochs 만큼 반복시키기.
for epoch in range(epochs):
    #enumerate로 인덱스와 (이미지 데이터, 타겟(=정답))을 가져옴. 이게 데이터의 형태임.
    for idx, (image, target) in enumerate(train_loader):
        #이미지와 타겟 디바이스로 전송
        image=image.to(device)
        target=target.to(device)
        
        #image를 모델에 넣어 출력값 만들기.
        out=model(image)
        
        #출력값과 타겟을 손실함수에 넣어 손실값 계산
        loss_value=loss(out,target)
        
        #남아있을 수 있는 기울기를 초기화.
        optim.zero_grad()
        
        #기울기 계산. 역전파.
        loss_value.backward()
        
        #파라미터 업데이트.
        optim.step()
        
        if idx%100 ==0:
            print(loss_value.item())