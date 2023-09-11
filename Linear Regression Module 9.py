#design model (input, output dorrward pass)
import torch 
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


#prepare data 
x_numpy,y_numpy=datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=1)

x=torch.from_numpy(x_numpy.astype(np.float32))
y=torch.from_numpy(y_numpy.astype(np.float32))

y=y.view(y.shape[0],1) #to reshape our tensor 

n_samples,n_features=x.shape

#model buidling
input_size=n_features
output_size=1
model=nn.Linear(input_size,output_size)

#loss and opt
learning_rate=0.01
crit=nn.MSELoss() #loss function
opt=torch.optim.SGD(model.parameters(),lr=learning_rate)

#training
n_epochs=200
for epoch in range(n_epochs):
    y_pred=model(x)
    loss=crit(y_pred,y)
    #backward pass
    loss.backward()
    #update weights 
    opt.step()

    opt.zero_grad()
    if(epoch%10==0):
        print(f'epoch: {epoch+1}, loss= {loss.item():.4f}')

#plotting
pred=model(x).detach().numpy() #detach as no need to use them in graph
plt.plot(x_numpy,y_numpy,'ro')
plt.plot(x_numpy,pred,'b')
plt.show()