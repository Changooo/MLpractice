import torchvision.transforms as transforms
from torchvision.datasets import MNIST, SVHN
import torch.utils.data as data_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader


mnist_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (1.0,))
])

# RGB -> GRAY 및 28 * 28 사이즈 변환
svhn_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.CenterCrop(28),
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (1.0,))
])

################################################################
download_root = './data'

train_mnist = MNIST(download_root, transform=mnist_transform, train=True, download=True)
test_mnist = MNIST(download_root, transform=mnist_transform, train=False, download=True)

svhn = SVHN(download_root, transform=svhn_transform, download=True)

# target domain 데이터 train 6만개, test 1만개 활용
train_indices = torch.arange(0, 60000)
test_indices = torch.arange(60000, 70000)
train_svhn = data_utils.Subset(svhn, train_indices)
test_svhn = data_utils.Subset(svhn, test_indices)

################################################################
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1)
        self.fc = nn.Linear(4 * 4 * 20, 100)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # (batch, 1, 28, 28) -> (batch, 10, 24, 24)

        x = F.max_pool2d(x, kernel_size=2, stride=2) # (batch, 10, 24, 24) -> (batch, 10, 12, 12)

        x = F.relu(self.conv2(x)) # (batch, 10, 12, 12) -> (batch, 20, 8, 8)

        x = F.max_pool2d(x, kernel_size=2, stride=2) # (batch, 20, 8, 8) -> (batch, 20, 4, 4)

        x = x.view(-1, 4 * 4 * 20) # (batch, 20, 4, 4) -> (batch, 320)

        x = F.relu(self.fc(x)) # (batch, 320) -> (batch, 100)
        return x # (batch, 100)

        ################################################################
        class GradReverse(torch.autograd.Function):
    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output): # 역전파 시에 gradient에 음수를 취함
        return (grad_output * -1)

class domain_classifier(nn.Module):
    def __init__(self):
        super(domain_classifier, self).__init__()
        self.fc1 = nn.Linear(100, 10)
        self.fc2 = nn.Linear(10, 1) # mnist = 0, svhn = 1 회귀 가정

    def forward(self, x):
        x = GradReverse.apply(x) # gradient reverse
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)

class label_classifier(nn.Module):
    def __init__(self):
        super(label_classifier, self).__init__()
        self.fc1 = nn.Linear(100, 25)
        self.fc2 = nn.Linear(25, 10) # class 개수 = 10개

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

###################################################################

class DANN_CNN(nn.Module):
    def __init__(self, CNN):
        super(DANN_CNN, self).__init__()

        self.cnn = CNN() # CNN 구조 모델 받아오기

        self.domain_classifier = domain_classifier() # 도메인 분류 layer

        self.label_classifier = label_classifier() # 숫자 0 ~ 9 클래스 분류 layer

    def forward(self, img):
        cnn_output = self.cnn(img) # (batch, 100)

        domain_logits =  self.domain_classifier(cnn_output) # (batch, 100) -> (batch, 1)

        label_logits = self.label_classifier(cnn_output) # (batch, 100) -> (batch, 10)

        return domain_logits, label_logits


###################################################################


class DANN_Loss(nn.Module):
    def __init__(self):
        super(DANN_Loss, self).__init__()

        self.CE = nn.CrossEntropyLoss() # 0~9 class 분류용
        self.BCE = nn.BCELoss() # 도메인 분류용
        
    # result : DANN_CNN에서 반환된 값
    # label : 숫자 0 ~ 9에 대한 라벨
    # domain_num : 0(mnist) or 1(svhn)
    def forward(self, result, label, domain_num, alpha = 1):
        domain_logits, label_logits = result # DANN_CNN의 결과

        batch_size = domain_logits.shape[0]

        domain_target = torch.FloatTensor([domain_num] * batch_size).unsqueeze(1).to(device)

        domain_loss = self.BCE(domain_logits, domain_target) # domain 분류 loss

        target_loss = self.CE(label_logits, label) # class 분류 loss

        loss = target_loss + alpha * domain_loss

        return loss


#########################################################

batch_size = 64

# dataloader 선언
mnist_loader = DataLoader(dataset=train_mnist, 
                         batch_size=batch_size,
                         shuffle=True)

svhn_loader = DataLoader(dataset=train_svhn, 
                         batch_size=batch_size,
                         shuffle=True)


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

my_cnn = CNN()

model = DANN_CNN(my_cnn).to(device)

loss_fn = DANN_Loss().to(device)

epochs = 10

model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs * len(mnist_loader))

alpha = 0.5

for i in range(1, epochs + 1):
    total_loss = 0

    for step in tqdm(range(len(mnist_loader))):

        # mnist, svhn에서 1 batch씩 가져오기
        source_data = iter(mnist_loader).next()
        target_data = iter(svhn_loader).next()
		
        # 각 batch 내 데이터 : 0번은 이미지 픽셀 값, 1번은 0 ~ 9 class 라벨 값
        mnist_data = source_data[0].to(device)
        mnist_target = source_data[1].to(device)

        svhn_data = target_data[0].to(device)
        svhn_target = target_data[1].to(device)

        # 순전파 결과 구하기
        source_result = model(mnist_data)
        target_result = model(svhn_data)

        # 순전파 결과, class label, domain label(0 = mnist, 1 = svhn), alpha 순서
        source_loss = loss_fn(source_result, mnist_target, 0, alpha = alpha)
        target_loss = loss_fn(target_result, svhn_target, 1, alpha = alpha)

        loss = source_loss + target_loss

        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    print('Epoch : %d, Avg Loss : %.4f'%(i, total_loss / len(mnist_loader)))