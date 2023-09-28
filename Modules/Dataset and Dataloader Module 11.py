# batch_size : no of training samples in one epoch
#no. of iterations=No. of passes , each pass using [batch_size number of samples]

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self): #data loading
        xy=np.loadtxt('D:\BDACC Learnings\Pytorch\PyTorch-DeepLearning\data\wine.csv',delimiter=",",dtype=np.float32,skiprows=1)
        self.x=torch.from_numpy(xy[:,1:])
        self.y=torch.from_numpy(xy[:,[0]])
        self.n_samples=xy.shape[0]

    def __getitem__(self,index):
        return self.x[index],self.y[index]

    def __len__(self):
        return self.n_samples 

dataset=WineDataset()
dataloader=DataLoader(dataset=dataset,batch_size=4,shuffle=True) #num_workers=2 is for usigng 2 cores of CPU but check first that it should be availabel

for data in dataloader:
    features,labels=data
    print(features,labels)

#dummy training
num_epochs=2
total_samples=len(dataset)
n_iterations=math.ceil(total_samples/4)
print(total_samples,n_iterations)  #178 samples and 45 iterations

#training loop
for epoch in range(num_epochs):
    for i,(inputs,lables) in enumerate(dataloader): #we use enumerate when need to keep track of index with the values : 1st arg is index and 2nd is value
        if (i+1)%5==0:
            print(f'epoch {epoch+1}/{num_epochs},step {i+1}/{n_iterations},inputs {inputs.shape}')

