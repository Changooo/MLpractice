import torch
import torch.nn as nn


class simpleMLP():
    def __init__(self, input_size, output_size):
        self.W = torch.FloatTensor(np.rand(input_size, output_size))
        self.b = torch.FloatTensor(np.rand(output_size))
        self.W.requires_grad_(True)
        self.b.requires_grad_(True)
    
    def forward(self, X):
        return X.matmul(self.W) + self.b
    
    def parameters(self):
        return [self.W, self.b]


class myCNN():
    def __init__(self, )

