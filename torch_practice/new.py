import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
import torch.optim as optim
device = torch.device("cuda")

training_epochs = 2
batch_size = 100
learning_rate = 0.1


train = pd.read_csv("train.csv")
test = pd.read_csv("testwithclass.csv")

train.drop("Time", axis=1, inplace=True)
test.drop("Time", axis=1, inplace=True)
# train.drop("Amount", axis=1, inplace=True)
# test.drop("Amount", axis=1, inplace=True)

data_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, drop_last=True)

class DeepClassifier(nn.Module):
  def __init__(self, numfeatures):
    super().__init__()
    self.numfeatures = numfeatures
    self.layer = nn.Sequential(
        nn.Linear(self.numfeatures, 20).to(device),
        nn.ReLU(),
        nn.Linear(20, 2).to(device)
    )
  
  def forward(self, x):
    return self.layer(x)

model = DeepClassifier()
optimizer = optim.ADAM(model.parameters(), lr=learning_rate)

for epoch in range(training_epochs):
  for batch in data_loader:
    x_train, y_train = batch
    print(x_train)
    print(y_train)
    prediction = model(x_train)
    print(prediction)
    cost = nn.CrossEntropyLoss()(prediction, y_train)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    exit(1)


