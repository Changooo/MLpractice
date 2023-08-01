import torch 
import torch.nn as nn
import torch.optim as optim


device = 'cuda'

x_train = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = torch.FloatTensor([[0], [1], [1], [0]])

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(2, 10)
        self.hidden1 = nn.Linear(10, 10)
        self.hidden2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        h1 = self.sigmoid(self.input(x))
        h2 = self.sigmoid(self.hidden1(h1))
        return self.sigmoid(self.hidden2(h2))
    
model = MLP()
optimizer = optim.SGD(model.parameters(), lr=1)

for epoch in range(2000):
    cost = nn.BCELoss()(model(x_train), y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
print(model(x_train))
    