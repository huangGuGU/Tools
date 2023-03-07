#########################################################################
# autograd.Variable是torch.autograd中很重要的类。
# 它用来包装Tensor，将Tensor转换为Variable之后，可以装载梯度信息。
#
#########################################################################
from 装饰器.decorator_程序启动 import logit
import torch


@logit
def computer_grad():
    x = torch.randn(2, 2)
    y = torch.randn(2, 2)
    z = torch.nn.Parameter(torch.randn(2, 2), requires_grad=True)

    a = x + y
    b = a + z
    out = b.mean()
    out.backward(retain_graph=True)

    print(x.requires_grad, y.requires_grad, z.requires_grad)  # False, False, True
    print(a.requires_grad, b.requires_grad)  # False, True
    print(z.grad)


if __name__ == '__main__':
    computer_grad()
    pass
