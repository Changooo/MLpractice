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


#  (모든 convolution 경우의 수, 각 채널 input의 convolution 타겟 나열) * (각 채널의 커널 나열, 다양한 커널)


class myCNN():
    def __init__(self, kernel_size, kernels, channels, width, height):
        self.W = torch.FloatTensor(np.rand(kernel_size*channels, kernels))
        self.b = torch.FloatTensor(np.rand(kernels)) 
        # 근데 이러면 각 채널들이 같은 편향을 가지게 되는데 괜찮나?
        self.kernel_size = kernel_size
        self.channels = channels
        self.width = width
        self.height = height

    def im2col(self, images):
        kernel_size = self.kernel_size
        channels = self.channels
        width = self.width
        height = self.height

        convolutions_in_a_row = ((width-kernel_size)//stride + 1)
        convolutions_in_a_col = ((height-kernel_size)//stride + 1)

        convolutions = convolutions_in_a_row*convolutions_in_a_col
        images_2col = np.array(convolutions, kernel_size*channels)
        for co in range(convolutions):
            for ch in range(channels):
                for kn in range(kernel_size**2):
                    index = stride*((co//convolutions_in_a_row)*height+(co%convolutions_in_a_row)) 
                    + (kn//kernel_size)*width + (kn%//kernel_size)
                    images_2col[co][ch*kernel_size+kn] = images[ch][index]
        return images_2col

    def reshape(self, output):
        # output = (convolutions, kernels)


    # images는 (channels, pixels)
    def forward(self, X):
        return self.im2col(X).matmul(self.W)+self.b
        # (kernels, convolutions) + (kernels) 이게 되나?
        
    def parameters(self):
        return [self.W, self.b]
    
    # 이렇게 펼쳐서 하면 역전파 어떻게 할거임?


