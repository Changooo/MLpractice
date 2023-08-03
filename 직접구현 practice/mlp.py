import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np

input_str = 'apxlea'
label_str = '1'
char_vocab = sorted(list(set(input_str+label_str)))
vocab_size = len(char_vocab)

x = torch.FloatTensor(np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[10,20]]))
y = torch.FloatTensor(np.array([5, 11, 17, 23, 29, 41]))

class simpleMLP:
    def __init__(self, input_size, output_size):
        self.W = torch.FloatTensor(np.random.rand(input_size, output_size))
        self.b = torch.FloatTensor(np.random.rand(output_size))
        # self.W = torch.FloatTensor([[2],[1]])
        # self.b = torch.FloatTensor([1])
        self.W.requires_grad_(True)
        self.b.requires_grad_(True)
    def forward(self, x):
        return x.matmul(self.W)+self.b    
    def parameters(self):
        return [self.W, self.b]
    
mlp = simpleMLP(2, 1)
optimizer = optim.Adam(mlp.parameters(), 0.1)
cost_function = torch.nn.MSELoss()
 
for epoch in range(1000):  
    cost = cost_function(mlp.forward(x), y.view(-1,1))
    if epoch % 1000 == 0:
        print(cost, mlp.parameters())
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

print(mlp.forward(torch.FloatTensor([9,1])))
