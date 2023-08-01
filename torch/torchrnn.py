import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

input_str = 'apxlea'
label_str = '1'
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


class Rnn(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size, bias=True)
        
    def forward(self, x):
        x, _status = self.rnn(x)
        x = x[...,-1:,:]
        x = self.fc(x)
        return x


input_size = vocab_size
hidden_size = 8
output_size = vocab_size
learning_rate = 0.01

rnn = Rnn(input_size, hidden_size, output_size)
optimizer = optim.Adam(rnn.parameters(), learning_rate)
cost_function = torch.nn.CrossEntropyLoss()

for epoch in range(1000):
    prediction = rnn(X)
    cost = cost_function(prediction.view(-1, output_size), Y.view(-1))
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    

print([index_to_char[i] for i in rnn(X).detach().numpy().squeeze(axis=0).argmax(axis=1)])