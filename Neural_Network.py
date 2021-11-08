from numpy.core.fromnumeric import shape
import torch
import os
import torch.nn as nn
#from torch.optim import optimizer
from torch.optim.optimizer import Optimizer
from torchvision import transforms
import torchvision
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math, random

# we are using torch libarary which includes mnist dataset and utility function for handling images.
#storing raw data into mnist_trainset
#trianset stores 60,000 images and testset stores 10,000 mnist dataset.
# what is transfor.toTesnor which is an operation from the torchvision libaray that does some preprocessing function

#====================MNIST Dataset========================================#
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())            
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

#storing trainset and test in data named file and root argument which is the path where the data is downloads 
# train is true means this function will return the trained dataset and train is false means it is testing dataset
print(mnist_trainset.data.max())

print('=============================')
print(mnist_trainset.data)
print(mnist_trainset.data.max())   #255 for white part or digits
print('===============================')
print(mnist_trainset.data.shape)        #28*28 image and 60k size
print('-===================================')
print(mnist_trainset.targets)
print('=====================================')


#defining model i.e Artificial Neural Network
model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2),
    nn.ReLU(),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
    nn.ReLU(),
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
    nn.ReLU()
)

criteria = nn.CrossEntropyLoss()    #meant for multiple categories
optimise = torch.optim.Adam(model.parameters())

batch_size = 64
#dataloader in pytorch
#create generator which allows us to loop through each batch of data.
train_loader = torch.utils.data.DataLoader(dataset=mnist_trainset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=mnist_testset, batch_size=batch_size, shuffle=False)

#shuffle data suffled in train dataset because for the trainig data if we loop over each samples in teh same order
# each time , this will introduce inter correlation which will decrease performance.
#  we wnat our sample to be random.
#testdata no need because we just need to evaluate the loss
   
mnist_trainset.transform(mnist_trainset.data.numpy()).max()

#step 4 : train our model 
# epoch is 10 very small, BGD, batch is a reps of entire sample, small subset of data, total no. of iteration is very high
# large no. of GD stes in for loop

n_epochs = 10
train_loss = np.zeros(n_epochs)
test_loss = np.zeros(n_epochs)

for i in range(n_epochs):
    train_loss1 = []
    # 2 for loop. loss for epcoh, but we get loss for each batch
    for inputs, targets in train_loader:
        inputs = inputs.view(-1, 28*28) #converting 2D into 1D vector reshaping 

        optimise.zero_grad()

        output = model(inputs)
        loss = criteria(output, targets)

        loss.backward()
        optimise.step()

        train_loss1.append(loss.item())

    train_loss1 = np.mean(train_loss1)

    test_loss1 = []
    for inputs, targets in train_loader:
        inputs = inputs.view(-1, 28*28)
        output = model(inputs)
        loss = criteria(output, targets)
        test_loss1.append(loss.item())
    test_loss1 = np.mean(test_loss1)
 
train_loss[i] = train_loss1
test_loss1[i] = test_loss1

print(f'Epoch {i+1}/{n_epochs}, Train Loss: {train_loss1 : 4f}, Test Loss : {test_loss1:4f}')


plt.plot(train_loss, label = 'train loss')
plt.plot(test_loss, label= 'test loss')
plt.show()

#n_correct = 0
#n_total = 0
# for inputs, targets in train_loader:
#     inputs = inputs.view(-1,784)
#     output = model(inputs)
#     _, predictions = torch.max(output,1)
#     n_correct = n_correct+ (predictions==targets).sum().item()
#     n_total += targets.shape[0]
# train_acc = n_correct/n_total

# print("==========================")
# print(len(train_loader))
# print("==========================")
# for x, y in test_loader:
#     print(x)   

# examples = iter(test_loader)
# example_data, example_targets = examples.next()
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.imshow(example_data[i][0], cmap='gray')
# plt.show()

#=============================Wireless======================#
# N = 10^5

# #transmitter
# num1 = 1
# num2 = 1
# # def randomInt(a,b):
# #     num = random.randint(0,1)
# #     if(num==0 and num1<b/2):
# #         num1 = 1
# #     elif(num==1 and num2<b/2):
# #         o = 1
# #     return num
    

# x = [] 
# for i in range(100000):
#     a = random.randint(0,1)
#     x.append(a)

# print(x[1])
# #BPSK Modulation  0->-1, 1->0
# s = np.zeros(100000)
# print(len(s))

# for i in range(100000):
#     s[i] = 2*x[i]- 1


# print(s)
# print(len(s))


# input = random.rantint(0,1)(1,10^6)>0.5
# print(input)