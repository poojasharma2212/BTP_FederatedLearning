import torch
import torch.nn as nn
import torch.nn.functional as Func
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
import numpy as np
from datetime import datetime
import torchvision
from torchvision import transforms,datasets
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# import logging
import os
import syft as sy



hook = sy.TorchHook(torch)
args = {
    'use_cuda' : True,
    'batch_size' : 64,
    'test_batch_size' : 1000,
    'lr' : 0.01,
    'log_interval' : 10,
    'epochs' : 10,
    'clients' : 10
}

clients = []

for i in range(args['clients']):
    clients.append({'hook': sy.VirtualWorker(hook, id="client{}".format(i+1))})

print(clients)
    

use_cuda = args['use_cuda'] and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#os.chdir("/content/drive/MyDrive/FL_ZaaPoo/data/MNIST/raw")

# ========== IID_Dataset ========== #

nUsers = 10
def mnistIID(data,nUsers):#this function randomly chooses 60k/10 (assuming 10 users) images and distributes them in iid fashion among the users.
    nImages=int(len(data)/nUsers)
    # print(num_images)
    usersDict,indices={},list(range(len(data))) #length of dataset is 60k
    for i in range(nUsers):
        np.random.seed(i) #starts with the same random number to maiantain similarity across runs
        #np.random.choice selects num_images number of random numbers from 0 to indices
        usersDict[i]=set(np.random.choice(indices,nImages,replace=False)) #set drops repeated items
        indices=list(set(indices)-usersDict[i])
        print("i :::", end=" ")
        print(i,usersDict[i])
        print("===========================================")
    return usersDict

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
#transform=transforms.ToTensor()
mnist_trainset = datasets.MNIST(root='./data', train=False, download=False, transform= transform)          
mnist_testset = datasets.MNIST(root='./data', train=False, download=False,transform= transform)
print(mnist_testset.data.max())
print(mnist_testset.data.shape)
print(mnist_trainset.targets)
k = len(set(mnist_testset.targets.numpy()))
print(k)
train_group=mnistIID(mnist_trainset,nUsers)
test_group=mnistIID(mnist_testset,nUsers)


class FedDataset(Dataset):#this class helps connect the random indices with the image+label container in the dataset
    def __init__(self,dataset,indx):
      self.dataset=dataset
      self.indx=[int(i) for i in indx]
        
    def __len__(self):
      return len(self.indx)
    
    def __getitem__(self,item):
      images,labels=self.dataset[self.indx[item]]
      return (torch.tensor(images),torch.tensor(labels))
    
    
def getImage(dataset,indices,batch_size):#load images using the class FedDataset
  return DataLoader(FedDataset(dataset,indices),batch_size=batch_size,shuffle=True)

for inx, client in enumerate(clients):
  trainset_id_list = list(train_group[inx]) 
  client['mnist_trainset'] = getImage(mnist_trainset, trainset_id_list, args['batch_size'])
  client['mnist_testset'] = getImage(mnist_testset, list(test_group[inx]), args['batch_size'])

  # print(inx, client['mnist_testset'].get())
  # print("---------------------------")
  # print(inx, client['mnist_trainset'].get())
  # print("===========================")
print("============================")
print(type(client['mnist_testset'])) 
print(type(client)) 
print("============================")
# ================================= #



class CNN(nn.Module):
  def __init__(self,k):  #constructor 
    super(CNN, self).__init__() # calling parent's class constructor
    self.conv_layers = nn.Sequential(     # Preparing Layers for the model followed by the ReLU function as the Activation function
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
        nn.ReLU()
    )

    # self.dense_layers = nn.Sequential(
    #     nn.Dropout(0.2),
    #     nn.Linear(128*2*2, 512),
    #     nn.ReLU(),
    #     nn.Dropout(0.2),
    #     nn.Linear(512, k)
    # )

    self.fc = nn.Sequential(
            nn.Linear(in_features=64*12*12, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10),
        )
    self.dropout = nn.Dropout2d(0.25)
    
  # def forward(self, X):
  #   out = self.conv_layers(X)
  #   out = out.view(out.size(0), -1)
  #   out = self.dense_layers(out)
  #   return out

  def forward(self, x):
        x = self.conv(x)
        x = Func.max_pool2d(x,2)
        x = x.view(-1, 64*12*12)
        x = self.fc(x)
        x = Func.log_softmax(x, dim=1)
        return x



model = CNN(k) 
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# model.to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters())
# batchSize = 128
# train_loader = torch.utils.data.DataLoader(dataset = mnist_trainset,
#                                            batch_size=batchSize,
#                                            shuffle=True)

# test_loader = torch.utils.data.DataLoader(dataset = mnist_testset,
#                                            batch_size=batchSize,
#                                            shuffle=False)

# def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs):
#  # raise NotImplementedError("Subclasses should implement this!")
#   train_losses = np.zeros(epochs)
#   test_losses = np.zeros(epochs)

#   for it in range(epochs):
#     t0 = datetime.now()
#     train_loss = []
#     for inputs, targets in train_loader:
#       inputs, targets = inputs.to(device), targets.to(device)  #moving data to GPU

#       optimizer.zero_grad() # set parameter gradient to zero

#       outputs = model(inputs)  # forward pass
#       loss = criterion(outputs, targets)

#       loss.backward()  #backward and optimize
#       optimizer.step()

#       train_loss.append(loss.item())
#     train_loss = np.mean(train_loss)
    
#     test_loss = []
#     for inputs, targets in test_loader:
#       inputs, targets = inputs.to(device), targets.to(device)
#       outputs = model(inputs)
#       loss = criterion(outputs, targets)
#       test_loss.append(loss.item())
#     test_loss = np.mean(test_loss)

#     train_losses[it] = train_loss
#     test_losses[it] = test_loss

#     dt = datetime.now() - t0

#     print(f'Epoch{it+1}/{epochs}, Train Loss: {train_loss: .4f}, \
#     Test Loss: {test_loss:.4f}, Duration: {dt}')

#   return train_losses, test_losses

      

# train_losses, test_losses = batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs=15)
# plt.plot(train_losses, label='train loss')
# plt.plot(test_losses, label='test loss')
# plt.legend()
# plt.show()


# n_correct = 0.
# n_total = 0.

# for inputs, targets in train_loader:
#   inputs, targets = inputs.to(device), targets.to(device) # moving data to GPU

#   outputs = model(inputs)

#   _, predictions = torch.max(outputs, 1) 

#   n_correct  = n_correct + (predictions==targets).sum().item()
#   n_total = n_total + targets.shape[0]

# train_acc = n_correct / n_total

# n_correct = 0.
# n_total = 0.

# for inputs, targets in test_loader:
#   inputs, targets = inputs.to(device), targets.to(device) # moving data to GPU

#   outputs = model(inputs)

#   _, predictions = torch.max(outputs, 1) 

#   n_correct  = n_correct + (predictions==targets).sum().item()
#   n_total = n_total + targets.shape[0]

# test_acc = n_correct / n_total


# print(f"Train acc: {train_acc: .4f}, Test acc: {test_acc: .4f}") 


