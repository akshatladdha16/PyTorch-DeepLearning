import torch
import torch.nn as nn

import numpy as np
from sklearn import datasets 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

bc=datasets.load_breast_cancer()
x,y=bc.data,bc.target
n_samples,n_features=x.shape
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1234,test_size=0.2)

# when dealing with logistic regression always do standard stcaling
sc=StandardScaler()
#only the data that we are going to feed in model
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

x_train=torch.from_numpy(x_train.astype(np.float32))
x_test=torch.from_numpy(x_test.astype(np.float32))
y_train=torch.from_numpy(y_train.astype(np.float32))
y_test=torch.from_numpy(y_test.astype(np.float32))

# y_train and y_test reshaping
y_train=y_train.view(y_train.shape[0],1) 
# view : pytorch method to reshape the tensor , first arg is to specifies the shape of tensor and 2nd is the valeu to reshape it 
y_test=y_test.view(y_test.shape[0],1)


class LogisticRegression(nn.Module):
    def __init__(self,n_input_features):
        super(LogisticRegression,self).__init__()
        #super Function: The super() function is used to access and call methods from the parent class. In this case, super(LogisticRegression, self).__init__() calls the constructor (__init__) of the parent class (nn.Module) and initializes it with any necessary configurations or attributes.
        #layers
        self.Linear=nn.Linear(n_input_features,1)
    
    #forward pass
    def forward(self,x):
        y_pred=torch.sigmoid(self.Linear(x))
        return y_pred
model=LogisticRegression(n_features)

#loss and optim
crit=nn.BCELoss() #binary cross entropy loss
learning_rate=0.01
optim=torch.optim.SGD(model.parameters(),lr=learning_rate)

num_epochs=200
for epochs in range(num_epochs):
    y_pred=model(x_train)
    loss=crit(y_pred,y_train)
    
    #backward pass
    loss.backward()
    # weights updates
    optim.step()

    #zero gradient again
    optim.zero_grad()

    if (epochs+1)%10==0:
        print(f' epoch:{epochs+1}, loss={loss.item():.4f}')

#evaluation
with torch.no_grad(): #to stop gradient computation for tensors while evaluation( as no need of that)
    # During the evaluation phase (when you're not training the model but using it to make predictions), you don't need to compute gradients. In fact, computing gradients during evaluation can be unnecessary and computationally expensive.
    y_pred=model(x_test)
    y_pred_cls=y_pred.round()
    acc=y_pred_cls.eq(y_test).sum()/float(y_test.shape[0])
    print(f' accuracy: {acc:.4f}')
    