import torch

x=torch.rand(3,requires_grad=True)
print(x)

y= x+2 #so this will create a computational graph that later will help in optimization (back propogation) to claculate gradients

# pytorch will automatically create back propogation function here with y and x
print(y) 

z= y*y*2
z=z.mean() #gradient func is mean backwards
print(z)

# now we will back propogate the z
z.backward() #this function is to calculate gradient and need to do mean operation before this to match the dimensions
print(x.grad)

# if not using mean op we need to add tensor of same dimesion
#v=torch.tensor([0.1,1.0,0.001],dtype=torch.float32)
#z.backward(v)
#print(x.grad)


# we can stop tracking the gradients by using
# method 1 : x.requires_grad_(False)
#method 2
y=x.detach() #so now y is same tensor without grad calculation
print(y)
#here we can see y does not have grad func

#method3 : with torch.no_grad():
with torch.no_grad():
    y=x+2
    print(y)


#dummy training example to understand the accumulation of gradients
weights=torch.ones(4,requires_grad=True)
for epoch in range(2):
    model_output=(weights*3).sum()
    model_output.backward()
    print(weights.grad) #with number of epochs increase , it starts accumulates the gradient values which is incorrect for training

    # to handle this we empty the grad before next operation
    weights.grad.zero_() #_ inplace operation before next epoch to empty gradient

#another way to perform optimization setps
optimizer=torch.optim.SGD(weights,lr=0.01) #lr is learning rate
optimizer.step() 
optimizer.zero_grad() #this is to empty the gradient values 


