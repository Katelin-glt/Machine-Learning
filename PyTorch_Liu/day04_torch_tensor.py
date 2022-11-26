'''
PyTorch实现反向传播
'''

import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0])
w.requires_grad = True   # 需要计算梯度

def forward(x):
    return x * w

def loss(x, y):    # 实质  构建计算图
    y_pred = forward(x)
    return (y_pred - y) ** 2

print("Predict (before training)", 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)   # 前馈。l为张量，计算时会构建计算图
        l.backward()    # 反馈。 自动把计算链路上的所有需要梯度的地方都求出来存到变量里面。只要一做backward，计算图就被释放
        print("\tgrad:", x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data  # 取data，不能使用张量计算，会生成计算图

        w.grad.data.zero_() # 不清零会累加
    print("Progress: ", epoch, l.item())
print("Predict (after training)", 4, forward(4).item())