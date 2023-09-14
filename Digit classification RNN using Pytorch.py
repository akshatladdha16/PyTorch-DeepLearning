# digit classification project on mnist dataset using pytorch
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

#device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
# input_size=784 #(28*28) mnist dataset contains images 28*28 pixels
hidden_size=128
num_classes=10
num_epochs=2
batch_size=100
learning_rate=0.001

input_size=28 #treating input as sequence of 28 pixels
sequence_size=28 #passing in sequence instead of single input
num_layers=2

# mnist dataset : Many to One architecture output side clssification task 
train_dataset=torchvision.datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)

test_dataset=torchvision.datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor())

#dataloader
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(RNN,self).__init__()
        self.num_layers=num_layers
        self.hidden_size=hidden_size
        self.rnn=nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
        #input shape= batch_size,seq_length,input_size
        self.fc=nn.Linear(hidden_size,num_classes) #fully connected layer for classification hidden layer

    def forward(self,x):
        # first is input and then initial hidden state
        h0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device) #x.size(0) : batch size

        out,_=self.rnn(x,h0)
        #shape of output : batch_size,seq_length,hidden_size
        # out(N,28,128) 128 hiddent state
        out=out[:,-1,:] #only last time step output with complete batch and features 
        out=self.fc(out)
        return out 


model=RNN(input_size,hidden_size,num_layers,num_classes).to(device)

#loss optimizers
crit=nn.CrossEntropyLoss()
opt=torch.optim.Adam(model.parameters(),lr=learning_rate)

#training
n_total_steps=len(train_loader)
for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        #reshape images (100,1,28*28)
        images=images.reshape(-1,sequence_size,input_size).to(device)
        labels=labels.to(device)

        #forward
        outputs=model(images)
        loss=crit(outputs,labels)

        #backward and optimize
        opt.zero_grad() #clearing the gradients
        loss.backward() #backpropagation
        opt.step() #updating the parameters

        if (i+1)%100==0:
            print(f'epoch {epoch+1}/{num_epochs},step {i+1}/{n_total_steps},loss={loss.item():.4f}')


#testing 
with torch.no_grad():
    n_correct=0
    n_samples=0
    for images,labels in test_loader:
        images=images.reshape(-1,sequence_size,input_size).to(device)
        labels=labels.to(device)
        outputs=model(images)
        #value,index
        _,predictions=torch.max(outputs.data,1)
        n_samples+=labels.size(0)
        n_correct+=(predictions==labels).sum().item()
    acc=100.0*n_correct/n_samples
    print(f'Accuracy of the network on the 1000 test images: {acc} %')


#GRU : Gated Recurrent Unit 
#LSTM : Long Short Term Memory(need initial self state for LSTM so need to add one c0 layer with h0 of same size) both are based on RNN implementation with some changes in code 