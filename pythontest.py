import torch
import numpy as np
a = torch.IntTensor(np.array([[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]]))
b = torch.IntTensor(np.array([[[2,3], [4,1], [2,3]], [[1,2], [9,1], [2,7]]]))

print(a.shape)
print(b.shape)
print(a.matmul(b))
print(a.matmul(b).shape)
