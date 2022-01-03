from numpy.core.numeric import indices
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as Func
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
import numpy as np
from datetime import datetime
import torchvision
from Dataset_FL import mnistnon_IID 
from torchvision import transforms,datasets
from torch.utils.data import DataLoader, Dataset
import logging
import os

# import Dataset
# from Dataset
import random
# import mat
import syft as sy

count = 0
args = {
    'batch_size' : 64,
    'test_batch_size' : 1000,
    'lr' : 0.03,
    'log_interval' : 10,
    'epochs' :2,
    'clients' : 20,
    'seed' : 0,
    'rounds' : 10,
    'C' : 0.9,
    'drop_rate' : 0.1,
    'images' : 60000,
    'datatype': 'non_iid',
    'use_cuda' : False,
    'save_model' : True
}


use_cuda = args['use_cuda'] and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

hook = sy.TorchHook(torch)
clients = []

for i in range(args['clients']):
    clients.append({'hook': sy.VirtualWorker(hook, id="client{}".format(i+1))})

# print(clients) 
#os.chdir("/content/drive/MyDrive/FL_ZaaPoo/data/MNIST/raw")

#****************** ========== IID_Dataset ========== ******************** #

def mnistIID(data,nUsers):
    nImages=int(len(data)/nUsers)
    # print(num_images)
    usersDict,indices={},[i for i in (range(len(data)))] #length of dataset is 60k
    for i in range(nUsers):
        np.random.seed(i) 
        usersDict[i]=set(np.random.choice(indices,nImages,replace=False)) 
        indices=list(set(indices)-usersDict[i])
        # print("i :::", end=" ")
    # print(usersDict)
    #print(len(usersDict), "-----------------")
    return usersDict

#************************ ======== Non-IID Dataset ========== ******************#
nuser = 20
def mnistnon_IID(data, nuser, test):
    clients = 50
    images = 1200
    if test :
        clients = 40
        images = 250
    client_index = [i for i in range(clients)]
    usersDict = {i:np.array([]) for i in range(nuser)}
    indices = np.arange(clients*images)
    print(indices)
    unsorted_label = data.train_labels.numpy()
    print(len(unsorted_label), "-----------")
    indices_unsorted = np.vstack((indices,unsorted_label))
    print("---*******")
    print(indices_unsorted)
    indices_label = indices_unsorted[:,indices_unsorted[1,:].argsort()]
    print(indices_label, "*********")
    indices = indices_label[0,:]
    print(indices, "0000")
    for i in range(nuser):
        np.random.seed(i)
        print(client_index, "-------")
        temp = set(np.random.choice(client_index, 2 ,replace=False))
        print(temp)
        client_index = list(set(client_index)- temp)
        for x in temp:
            usersDict[i] = np.concatenate((usersDict[i], indices[x*images:(x+1)*images]), axis=0)
            # print(usersDict)
    return usersDict

nUsers = 20
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
#transform=transforms.ToTensor()
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform= transform)          
mnist_testset = datasets.MNIST(root='./data', train=False, download=True,transform= transform)
#print(mnist_testset.data.max())
# print(mnist_testset.data.shape)
#print(mnist_trainset.targets)
# print(mnist_testset)
k = len(set(mnist_testset.targets.numpy()))
# print(k)
if(args['datatype'] == 'iid'):
    train_group=mnistIID(mnist_trainset,nUsers) #dictionary containing dictionary for 20 clients 
    test_group=mnistIID(mnist_testset,nUsers)
    print(len(train_group[1]))
    print(len(test_group[1]))
elif(args['datatype'] == 'non_iid'):
    train_group=mnistnon_IID(mnist_trainset,nUsers, False)
    test_group=mnistnon_IID(mnist_testset,nUsers, True)
    print(len(train_group[1]))
    print(len(test_group[1]))


class FedDataset(Dataset):
    def __init__(self,dataset,indx):
      self.dataset=dataset
      self.indx=[int(i) for i in indx]
        
    def __len__(self):
      return len(self.indx)
    
    def __getitem__(self,item):
      images,labels=self.dataset[self.indx[item]]
      return (torch.tensor(images).clone().detach(),torch.tensor(labels).clone().detach())
    
    
def getImage(dataset,indices,batch_size):#load images using the class FedDataset
  return DataLoader(FedDataset(dataset,indices),batch_size=batch_size,shuffle=True)

for inx, client in enumerate(clients):
  trainset_id_list = list(train_group[inx]) 
  client['mnist_trainset'] = getImage(mnist_trainset, trainset_id_list, args['batch_size'])
  client['mnist_testset'] = getImage(mnist_testset, list(test_group[inx]), args['batch_size'])
  client['samples'] = len(trainset_id_list)/args['images']
  print(client['mnist_trainset'])

print("==================================")
# for inx, client in enumerate(clients):
  # client['mnist_testset'] = getImage(mnist_testset, list(test_group[inx]), args['batch_size'])
  # client['samples'] = len(trainset_id_list)/args['images']
  # print(client['mnist_testset'])
  # print(inx, client['mnist_testset'])
  # print("---------------------------")
  # print(inx, client['mnist_trainset'])
#   print("===========================")
# print("============================")
# print(type(client['mnist_testset'])) 
# print(type(client)) 
# print("============================")
# ================================= #

#=================Global Model===================#
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
global_test_dataset = datasets.MNIST('./', train=False, download=True, transform=transform)
global_test_loader = DataLoader(global_test_dataset, batch_size=args['batch_size'], shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = Func.relu(self.conv1(x))
        x = Func.max_pool2d(x, 2, 2)
        x = Func.relu(self.conv2(x))
        x = Func.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = Func.relu(self.fc1(x))
        x = self.fc2(x)
        return Func.log_softmax(x, dim=1)
    

def train(args, client, device):
    client['model'].train()
    client['model'].send(client['hook'])
    # print(client)
    # iterate over federated data
    for epoch in range(1,args['epochs']+1):
      for batch_idx, (data, target) in enumerate(client['mnist_trainset']):
        data = data.send(client['hook'])
        target = target.send(client['hook'])
        data, target = data.to(device), target.to(device)
        client['optimizer'].zero_grad()
        output = client['model'](data)
        loss = Func.nll_loss(output, target)
        loss.backward()
        client['optimizer'].step()
        # cli['optimizer'].zero_grad()
        # optimizer.step()
        
 
        if batch_idx % args['log_interval'] == 0:
            loss = loss.get()
            # print(loss.item())
            print(' Model  {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    client['hook'].id, epoch,
                    batch_idx * args['batch_size'], # no of images done
                    len(client['mnist_trainset']) * args['batch_size'], # total images left
                    100. * batch_idx / len(client['mnist_trainset']), 
                    loss.item()
                )
            )
            # print('Model {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #             client['hook'].id,
            #             epoch, batch_idx * args['batch_size'], len(client['mnist_trainset']) * args['batch_size'], 
            #             100. * batch_idx / len(client['mnist_trainset']), loss.item()))
    client['model'].get()
accu = []
def test(args,model, device, test_loader, count):
    print(count+1)
    count =  count+1
    print("TEST SET PRDEICTION")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # add losses together
            test_loss += Func.nll_loss(output, target, reduction='sum').item() 

            # get the index of the max probability class
            pred = output.argmax(1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss for model: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    accu.append(100. * correct / len(test_loader.dataset))

# model = CNN(k)
#optimizer = optim.SGD(model.parameters(), lr=args['lr'])

logging.info("Starting training !!")

torch.manual_seed(args['seed'])
global_model = CNN() 

for client in clients:
        torch.manual_seed(args['seed'])
        client['model'] = CNN().to(device)
        # print(client)
        client['optimizer'] = optim.SGD(client['model'].parameters(), lr=args['lr'])
        
# print(client)
def averageModels(global_model, clients):
      client_models = [clients[i]['model'] for i in range(len(clients))]
      samples = [clients[i]['samples'] for i in range(len(clients))]
      global_dict = global_model.state_dict()
    
      for k in global_dict.keys(): #key is CNN layer index and value is layer parameters
          global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() * samples[i] for i in range(len(client_models))], 0).sum(0) #take a weighted average and not average because the clients may not have the same amount of data to train upon
            
      global_model.load_state_dict(global_dict)
      return global_model

for fed_round in range(args['rounds']):
    
    # number of selected clients
    m = int(max(args['C'] * args['clients'], 1)) #at least 1 client is selected for training

    # Selected devices
    np.random.seed(fed_round)
    selected_clients_inds = np.random.choice(range(len(clients)), m, replace=False)#dont choose same client more than once
    selected_clients = [clients[i] for i in selected_clients_inds]
    
    # Active devices
    np.random.seed(fed_round)
    active_clients_inds = np.random.choice(selected_clients_inds, int((1-args['drop_rate']) * m), replace=False) #drop clients
    active_clients = [clients[i] for i in active_clients_inds]
    
    # Training 
    # print(client)
    # print('=============\\\\\\\=====================')
    for client in active_clients:
        # print(client)
        train(args,client, device)
        # print(client)
    
#     # Testing 
#     for client in active_clients:
#         test(args, client['model'], device, client['testset'], client['hook'].id)

    # Averaging 
    global_model = averageModels(global_model, active_clients)
    
    # Testing the average model
    test(args,global_model, device, global_test_loader, count)
          
    for client in clients:
        client['model'].load_state_dict(global_model.state_dict())
        
if (args['save_model']):
    torch.save(global_model.state_dict(), "FederatedLearning.pt")


print("============ Accuracy ===========")
print(accu)



