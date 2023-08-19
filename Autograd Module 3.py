import torch

x=torch.rand(3,requires_grad=True)
print(x)

y= x+2 #so this will create a computational graph that later will help in optimization (back propogation)
