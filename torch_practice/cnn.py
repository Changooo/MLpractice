import torch
import torch.nn as nn

inputs = torch.Tensor(1, 1, 28, 28)

conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
# 1 channel -> 32 channel (use 32 kernels)
# kernel size = 3*3, padding size = 1

conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
# 32 channel -> 64 channel (use 2 kernels)
# kernel size = 3*3, padding size = 1

pool = nn.MaxPool2d(2)
# kernel size = 2*2, stride = 2

out = conv1(inputs)
out = pool(out)
out = conv2(out)
out = pool(out)

out = out.view(out.size(0), -1)

fc = nn.Linear(3136,10)
out = fc(out)
print(out.shape)



