import torch
import torch.nn as nn
class LinearRegression(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LinearRegression,self).__init__()

        #define layers
        self.lin=nn.Linear(input_dim,output_dim)
    def forward(self,x):
        return self.lin(x)


