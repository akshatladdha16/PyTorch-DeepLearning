import torch 
import numpy as np
# numpy from tensor
a=torch.ones(5)
print(a)
b=a.numpy()
print(b)

#chnge in one of them will reflect the other also for eg
a.add_(1)
print(a)
print(b) #it will chnage b also 

#tensor from numpy
a=np.ones(6)
print(a)
b=torch.from_numpy(a)
print(b)


#Note numpy can only handle CPU operation not GPU.

if torch.cuda.is_available():
    print("Hello")

x=torch.ones(5,requires_grad=True) #this will tell tensor to calculate gradient later when using optimizers to optimize this variable x
print(x)




