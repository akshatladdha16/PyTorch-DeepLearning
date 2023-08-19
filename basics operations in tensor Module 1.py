import torch 
# everything based on tensors : 1d,2d, or any dimesnion
x=torch.rand(3) # to create a random tensor

x=torch.empty(3) #empty tensor 
x=torch.ones(2,2,dtype=torch.float16) # datatype assigned as float16

y=torch.rand(2,2)
x=torch.rand(2,2)

#adding function 
y.add_(x) #any function having _ will do inplace operation , here y will be updated with x+y values

z=x-y 
z=torch.sub(x,y) #for multiplication torch.mul(x,y) , no inplace operation


#slicing operation in tensor same as numpy
x=torch.rand(5,3)
print(x)
print(x[1,:])# 2nd row all columns
print(x[:,2]) # complete 3 column  

#reshaping op.
x=torch.rand(4,4)
y=x.view(16)#here our x was 5*3 so need to give same size here also
z=x.view(-1,8) # automatically adjust to x size with 8 as one of the row or col
print(x)
print(y.size()) 