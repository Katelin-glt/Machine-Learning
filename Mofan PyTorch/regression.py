import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linespace(-1, 1, 100), dim=1)
# torch中数据是有维度的，[1,2,3,4]为一维，[[1,2,3,4]]为2维，
y = x.pow(2) + 0.2*torch.rand(x.size()) # 加一点噪点的影响

x, y = Variable(x), Variable(y)
