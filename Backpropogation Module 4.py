#chain rule concept used in backpropogation
import torch
x=torch.tensor(1.0)
y=torch.tensor(2.0)

w=torch.tensor(1.0,requires_grad=True) #weights initialization

# forward pass and loss computation
y_hat=w*x
loss=(y_hat-y)**2 #loss function RMS

#backward pass
loss.backward()
print(w.grad)

#update weights
#next forward and backward pass