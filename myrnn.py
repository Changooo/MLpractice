import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np

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



input_str = 'apxlea'
label_str = '6'
char_vocab = sorted(list(set(input_str+label_str)))
vocab_size = len(char_vocab)

char_to_index = dict((c, i) for i, c in enumerate(char_vocab)) # 문자에 고유한 정수 인덱스 부여
index_to_char={}
for key, value in char_to_index.items():
    index_to_char[value] = key
    
    
x_data = [[char_to_index[c] for c in input_str]]
y_data = [[char_to_index[c] for c in label_str]]
x_one_hot = [np.eye(vocab_size)[i] for i in x_data]

X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)

input_size = vocab_size
hidden_size = 8
output_size = vocab_size
output_length = len(label_str)
learning_rate = 0.01        
    
myrnn = myRNN(input_size, hidden_size, output_size, output_length)
optimizer = optim.Adam(myrnn.parameters(), learning_rate)
cost_function = torch.nn.CrossEntropyLoss()
 
for epoch in range(1000):  
    prediction = myrnn.forward(X)
    cost = cost_function(prediction.view(-1, output_size), Y.view(-1))
    if epoch % 1000 == 0:
        pass
        # print(cost, mlp.parameters())
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

print([index_to_char[i] for i in myrnn.forward(X).detach().numpy().squeeze(axis=0).argmax(axis=1)])