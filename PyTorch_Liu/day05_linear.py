'''
PyTorch实现线性回归
'''
import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)


def func(*args, **kwargs):
    print(args)   # 元组
    print(kwargs)  # 字典

# func(1, 2, 4, 3, x=3, y=5)

# class Foobar:
#     def __init__(self):
#         pass
#
#     def __call__(self, *args, **kwargs):
#         forward(self, )
#         print("Hello" + str(args[0]))
#
# foobar = Foobar()
# foobar(1, 2, 3)