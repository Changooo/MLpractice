import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader 

import matplotlib.pyplot as plt
import csv

class simpleMLP:
    def __init__(self, input_size, output_size):
        self.W = torch.FloatTensor(np.random.rand(input_size, output_size))
        self.b = torch.FloatTensor(np.random.rand(output_size))
        self.W.requires_grad_(True)
        self.b.requires_grad_(True)
        
    def forward(self, x):
        return x.matmul(self.W)+self.b    
    
    def parameters(self):
        return [self.W, self.b]
    
class simpleRNN:
    def __init__(self, input_size, hidden_size):
        self.cell = simpleMLP(input_size, hidden_size)
        self.Wh = torch.FloatTensor(np.random.rand(hidden_size, hidden_size))
        self.bh = torch.FloatTensor(np.random.rand(hidden_size))
        self.status = None
        self.output = None
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        for t in range(x.shape[-2]):
            if t==0:
                self.status = self.cell.forward(x[..., t:t+1, :])
                self.status = self.tanh(self.status)
                self.output = self.status
            else:
                self.status = self.cell.forward(x[..., t:t+1, :]) + self.status.matmul(self.Wh)+self.bh
                self.status = self.tanh(self.status)
                self.output = torch.cat([self.output, self.status], dim=-2)
        return self.output
    
    def parameters(self):
        return [*self.cell.parameters(), self.Wh, self.bh]
    
class myRNN:
    def __init__(self, input_size, hidden_size, output_size, output_length):
        self.output_length = output_length
        self.rnn = simpleRNN(input_size, hidden_size)
        self.mlp = simpleMLP(hidden_size, output_size)
    
    def forward(self, x):
        rnn_output = self.rnn.forward(x)
        rnn_output = rnn_output[..., -self.output_length:, :]
        return self.mlp.forward(rnn_output)
    
    def parameters(self):
        return [*self.rnn.parameters(), *self.mlp.parameters()]

def preprocessing(raw_data, window_size, output_size):
    x_rain = [float(rain) if rain!='-' else -1.0 for time, level, rain, acc in raw_data]
    y_level = [float(level) if level!='-' else -1.0 for time, level, rain, acc in raw_data]
    
    dataset = list(zip(x_rain, y_level))
    x_data = []
    y_data = []
    for index in range(len(dataset)):
        if index+window_size+output_size > len(dataset):
            break
        x_data.append([[rain, level] for rain, level in dataset[index:index+window_size]])
        y_data.append([[level] for rain, level in dataset[index+window_size:index+window_size+output_size]])
    
    return torch.FloatTensor(x_data), torch.FloatTensor(y_data)

def split_train_and_test(preprocessed_data, percentage):
    x, y = preprocessed_data
    split = int(len(x)*percentage)
    x_train = x[:split, ...]
    y_train = y[:split, ...]
    x_test = x[split:, ...]
    y_test = y[split:, ...]
    return (x_train, y_train), (x_test, y_test)


# gpu device
#device = torch.device("cuda")


# features
input_size = 2
output_size = 1

# hyper parameters
window_size = 30
output_length = 10
hidden_size = 3
learning_rate = 0.1  
train_size = 0.99
batch_size = 90000
epoch = 400

# load data
raw_data = []
f = open('data.csv', 'r')
reader = csv.reader(f)
for i, row in enumerate(reader):
    if i==0:
        continue
    raw_row=[]
    for item in row:
        raw_row.append(item)
    raw_data.append(raw_row)
f.close() 

# preprocess data
preprocessed_data = preprocessing(raw_data, window_size, output_length)   
train_data, test_data = split_train_and_test(preprocessed_data, train_size)
x_train, y_train = train_data
dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# model set
myrnn = myRNN(input_size, hidden_size, output_size, output_length)
optimizer = optim.Adam(myrnn.parameters(), learning_rate)
cost_function = nn.MSELoss()

    
# train data
for epoch in range(epoch):
    for index, minibatch in enumerate(dataloader):
        #droplast
        if index == len(dataloader)-1:
            break
        
        X, Y = minibatch
        prediction = myrnn.forward(X)
        cost = cost_function(prediction, Y)
    
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        print(cost)

# prediction = myrnn.forward(x_train[2197:2198, ...][0])
# print(prediction)
# print(y_train[2197])

# test data
x_test, y_test = test_data
dataset = TensorDataset(x_test, y_test)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

predict_after10min = []
real_after10min = []
for index, minibatch in enumerate(dataloader):
    X, Y = minibatch
    prediction = myrnn.forward(X)
    predict_after10min.append(prediction.squeeze(-1).squeeze(0).detach().numpy()[-1])
    if index % 100 == 0:
        print(X)
        print(prediction)
    real_after10min.append(Y.squeeze(-1).squeeze(0).numpy()[-1])

plt.plot(predict_after10min, label="predict")
plt.plot(real_after10min, label="real")
plt.show()