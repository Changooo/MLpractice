import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import torch.optim as optim
import torch.nn.functional as F
device = torch.device("cuda")

training_epochs = 2
batch_size = 100
learning_rate = 0.1


mnist_train = dsets.MNIST(root='./MNIST_data', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root='./MNIST_data', train=False, transform=transforms.ToTensor(), download=True)

data_loader = DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)

class Softmax(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(784, 10).to(device)
  
  def forward(self, x):
    return self.linear(x)

model = Softmax()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(training_epochs):
  for batch in data_loader:
    x_train, y_train = batch
    x_train = x_train.view([-1, 784]).to(device)
    y_train = y_train.to(device)
    prediction = model(x_train)
    cost = nn.CrossEntropyLoss()(prediction, y_train)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()


r = 5
test_picture = mnist_test[r][0].view([1, 784]).to(device)
print(torch.argmax(F.softmax(model(test_picture), dim=1), 1))
plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
plt.show()