import torch
import torch.nn as nn


class simpleMLP:
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
    def __init__(self, width, height, kernel_size, stride, input_kernels, output_kernels):
        convolutions_in_a_row = ((width-kernel_size)//stride + 1)
        raw_kernel = np.rand(kernel_size**2)
        kernel = np.zeros((width*height, convolutions_in_a_row**2))
        for i in range(convolutions_in_a_row**2):
            for j in range(kernel_size**2):
                index = stride*((i//convolutions_in_a_row)*width+(i%convolutions_in_a_row)) 
                + (j//kernel_size)*width + (j%//kernel_size)
                kernel[index][i] = raw_kernel[j]

        self.kernel = torch.FloatTensor(kernel)
        self.bias = torch.FloatTensor(np.rand(kernel_size)**2)
        self.convolutions_in_a_row = convolutions_in_a_row
        self.kernel_size = kernel_size
        self.stride = stride
        self.width = width
        self.height = height

    def forward(self, X):
        output = X.matmul(self.kernel)
        for i in range(self.convolutions_in_a_row**2):
            for j in range(self.kernel_size**2):
                index = stride*((i//convolutions_in_a_row)*width+(i%convolutions_in_a_row)) 
                + (j//kernel_size)*width + (j%//kernel_size)
                output[index][i] += self.bias[j]
        return output
        
    def parameters(self):
        return [self.kernel, self.bias]
    
    # 이렇게 펼쳐서 하면 역전파 어떻게 할거임?


