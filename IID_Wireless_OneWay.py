import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as Func
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
import numpy as np
from datetime import datetime
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import logging
import os
import random
import math
import syft as sy
from funtions import mnistIID, mnistnon_IID, FedDataset, getImage
from utils import averageModels

Ps = 2  # signal power
key = []
for i in range(60000):  # generating a random password to activate training (Pilot signal)
    temp = random.randint(0, 1)
    key.append(temp)

key_n = [0]*len(key)
for i in range(len(key)):  # bpsk modulation
    if(key[i] == 1):
        # print("yay")
        key_n[i] = -math.sqrt(Ps)
    else:
        key_n[i] = math.sqrt(Ps)

# print(key)

key_array = np.array(key_n)

accu = []


def Wrapper(batch_size, lr, no_of_epoch, no_of_clients, no_of_rounds, key, key_array, Ps):
    count = 0
    print("yes")
    args = {
        'batch_size': 64,
        'test_batch_size': 1000,
        'lr': 0.01,
        'log_interval': 10,
        'epochs': 3,
        'clients': 20,
        'seed': 0,
        'rounds': 20,
        'C': 0.9,
        'lowest_snr': 0,
        'highest_snr': 38,
        'lowest_csi': 0,
        'highest_csi': 1,
        'drop_rate': 0.1,
        'images': 60000,
        'datatype': 'iid',
        'use_cuda': False,
        'save_model': True
    }

    use_cuda = args['use_cuda'] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    #kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    hook = sy.TorchHook(torch)
    clients = []

    for i in range(args['clients']):
        clients.append({'hook': sy.VirtualWorker(
            hook, id="client{}".format(i+1))})

    # print(clients)
    # os.chdir("/content/drive/MyDrive/FL_ZaaPoo/data/MNIST/raw")
    nUsers = 20
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # transform=transforms.ToTensor()
    mnist_trainset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    mnist_testset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    # print(mnist_testset.data.max())
    # print(mnist_testset.data.shape)
    # print(mnist_trainset.targets)
    # print(mnist_testset)
    k = len(set(mnist_testset.targets.numpy()))
    # print(k)
    if(args['datatype'] == 'iid'):
        print("iid")
        # dictionary containing dictionary for 20 clients
        train_group = mnistIID(mnist_trainset, nUsers)
        test_group = mnistIID(mnist_testset, nUsers)
        print(len(train_group[1]))
        print(len(test_group[1]))
    elif(args['datatype'] == 'non_iid'):
        print("non_iid")
        train_group = mnistnon_IID(mnist_trainset, nUsers)
        test_group = mnistnon_IID(mnist_testset, nUsers)
        print(len(train_group[1]))
        # print(len(test_group[1]))

    for inx, client in enumerate(clients):
        trainset_id_list = list(train_group[inx])
        client['mnist_trainset'] = getImage(
            mnist_trainset, trainset_id_list, args['batch_size'])
        client['mnist_testset'] = getImage(
            mnist_testset, list(test_group[inx]), args['batch_size'])
        client['samples'] = len(trainset_id_list)/args['images']
    # print(client['mnist_trainset'])

    # print("================yyyyyyy==================")
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
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    global_test_dataset = datasets.MNIST(
        './', train=False, download=True, transform=transform)
    global_test_loader = DataLoader(
        global_test_dataset, batch_size=args['batch_size'], shuffle=True)

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
        Client_Status = False
        client['model'].train()
        # if(csi==0 or mu==0):
        # Optimal_Power = 0
        # else:
        # Optimal_Power = max(0,(1/mu- 1/csi))
        # Optimal_Power = max(0,(1/mu - 1/csi))
        # print("Optimal power allocated is: ", Optimal_Power)

        # snr_val = 10**(snr/10)

        # absh = csi*Optimal_Power/snr_val
        # x=random.uniform(0,absh)
        # #std = math.sqrt(Ps/snr_val)
        # #y = random.random()
        # y=math.sqrt(absh*absh-x*x)
        # std=math.sqrt(Optimal_Power/snr_val*absh*absh)

        # h = complex(x,y)
        # print("snr in dB",snr )

        # if(Optimal_Power!= 0):

        #     data= client['model'].conv1.weight
        #     data = data*math.sqrt(Optimal_Power)
        #     noise = torch.randn(data.size())
        #     y_out = h*data + noise*std
        #     y_out = y_out/(math.sqrt(Optimal_Power)*(h))
        #     y_out = y_out.real

        #     client['model'].conv1.weight.data = y_out

        #     data= client['model'].conv2.weight
        #     data = data*math.sqrt(Optimal_Power)
        #     noise = torch.randn(data.size())
        #     y_out = h*data + noise*std
        #     y_out = y_out/(math.sqrt(Optimal_Power)*(h))
        #     y_out = y_out.real

        #     client['model'].conv2.weight.data = y_out

        client['model'].send(client['hook'])
        print("Client:", client['hook'].id)

        # print("CSI",csi)
        # print()

        # key_received = h*key_array+(np.random.randn(len(key_array))*std*2)
        # #print(key_array_received)
        # key_received=(key_received/(h)).real

        # for n in range (len(key_received)):
        #     if(key_received[n]>=0):
        #         key_received[n]=0
        #     else:
        #         key_received[n]=1

        # key_received=key_received.tolist()
        # key_received = [int(item) for item in key_received]

        # # print(client)
        # # iterate over federated data

        # Xor_sum = sum(np.bitwise_xor(key,key_received))
        # error = Xor_sum/len(key)
        error = 0
        # if(error == 0 and Optimal_Power >0):
        if(error == 0):
            Client_Status = True
            for epoch in range(1, args['epochs']+1):
                for batch_idx, (data, target) in enumerate(client['mnist_trainset']):
                    data = data.send(client['hook'])
                    target = target.send(client['hook'])
                    data, target = data.to(device), target.to(device)
                    client['optimizer'].zero_grad()
                    output = client['model'](data)
                    loss = Func.nll_loss(output, target)
                    loss.backward()
                    # print(loss.grad)
                    client['optimizer'].step()
                    # cli['optimizer'].zero_grad()
                    # optimizer.step()

                    #print("==========ye chalega kya========================")
                    if batch_idx % args['log_interval'] == 0:
                        loss = loss.get()
                        # print(loss.item())
                        # print(' Model  {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        #         client['hook'].id, epoch,
                        #         batch_idx * args['batch_size'], # no of images done
                        #         len(client['mnist_trainset']) * args['batch_size'], # total images left
                        #         100. * batch_idx / len(client['mnist_trainset']),
                        #         loss.item()
                        #     )
                        # )
                        print('Model {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            client['hook'].id,
                            epoch, batch_idx *
                            args['batch_size'], len(
                                client['mnist_trainset']) * args['batch_size'],
                            100. * batch_idx / len(client['mnist_trainset']), loss.item()))
        # else:
        #    print("Channel is not taken for fedavg in this round")
        client['model'].get()

        # return Client_Status, Optimal_Power

    # def ClientUpdateVal(clients,key,key_array,power):
    #     good_channel =[]
    #     for client in clients:
    #         snr=random.randint(args['lowest_snr'],args['highest_snr'])
    #         print("snr in dB = ",snr)
    #         print("Client:",client['hook'].id)
    #         snr_value=10**(snr/10)
    #         std=math.sqrt(Ps/snr_value) #channel noise
    #         x=random.random()
    #         y=random.random()
    #         h=complex(x,y)

    #         data=client['model'].conv1.weight
    #         data=data*math.sqrt(Ps)

    #         power += torch.norm(abs(data*data)).item()
    #         noise = (torch.randn(data.size())*std)
    #         y_out = h*data + noise
    #         y_out = y_out/(math.sqrt(Ps)*(h))
    #         y_out= y_out.real
    #         client['model'].conv1.weight.data=y_out

    #         data=client['model'].conv2.weight
    #         data=data*math.sqrt(Ps)
    #         noise1 = (torch.randn(data.size())*std)
    #         y_out = h*data + noise1
    #         y_out = y_out/(math.sqrt(Ps)*(h))
    #         y_out= y_out.real
    #         client['model'].conv2.weight.data=y_out

    #         key_received=h*key_array+(np.random.randn(len(key_array))*std*2)
    #         key_received=(key_received/(h)).real

    #         for n in range (len(key_received)):
    #             if(key_received[n]>=0):
    #                 key_received[n]=0
    #             else:
    #                 key_received[n]=1

    #         key_received=key_received.tolist()
    #         key_received = [int(item) for item in key_received]

    #         Xor_sum = sum(np.bitwise_xor(key_received,key))
    #         error = Xor_sum/len(key)
    #         if(error == 0):
    #             print("This is a Good Channel")
    #             good_channel.append(client)
    #         else:
    #             print("This is a Poor Channel")
    #         print()

    #     return good_channel, power

    def test(args, model, device, test_loader, count):
        # print("TEST SET PRDEICTION")
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                # add losses together
                test_loss += Func.nll_loss(output,
                                           target, reduction='sum').item()

                # get the index of the max probability class
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss for model: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        accu.append(100. * correct / len(test_loader.dataset))

        print('=====accu======', accu)
    # model = CNN(k)
    #optimizer = optim.SGD(model.parameters(), lr=args['lr'])

    logging.info("Starting training !!")

    torch.manual_seed(args['seed'])
    global_model = CNN()

    for client in clients:
        torch.manual_seed(args['seed'])
        client['model'] = CNN().to(device)
        # print(client)
        client['optimizer'] = optim.SGD(
            client['model'].parameters(), lr=args['lr'])

    # print(client)

    for fed_round in range(args['rounds']):
        # print("fed_round")
        # number of selected clients
        client_good_channel = []

        # at least 1 client is selected for training
        m = int(max(args['C'] * args['clients'], 1))

        # Selected devices
        np.random.seed(fed_round)
        # dont choose same client more than once
        selected_clients_inds = np.random.choice(
            range(len(clients)), m, replace=False)
        selected_clients = [clients[i] for i in selected_clients_inds]

        # Active devices
        np.random.seed(fed_round)
        active_clients_inds = np.random.choice(selected_clients_inds, int(
            (1-args['drop_rate']) * m), replace=False)  # drop clients
        active_clients = [clients[i] for i in active_clients_inds]

        # Training
        # print(client)

        # snr = []
        # csi = []

        # for i in range(args['clients']-1):
        #     # print("csi")
        #     csi.append(random.uniform(args['lowest_csi'], args['highest_csi']))
        #     snr.append(random.randint(args['lowest_snr'], args['highest_snr']))

        #    # snr.append(random.randint())
        # print("CSI=========>>>>>>>>>>>>>",csi)
        # #===============Water Filling Algorithm ==============
        # mu_min = 1e-15
        # mu = 0

        # wfa = 3.402823466E+38

        # while(mu_min<=1):
        #     wfa1 = 0
        #     P_total = 0

        #     for csi_i in csi:
        #         # print(csi_i)
        #         if(csi_i==0 or mu_min==0):
        #             P_optimal = 0
        #         else:
        #             P_optimal = max(0,(1/mu_min - 1/csi_i))
        #         wfa1 = math.log( 1+ P_optimal**csi_i)
        #         P_total += P_optimal

        #     len_ac = len(active_clients)
        #     g = wfa1 - mu_min* (P_total - Ps*(len_ac-1))
        #     if(g<wfa):
        #         mu = mu_min
        #         wfa = g
        #     print("mu ki value", mu)
        #     mu_min+= 0.00004

        # print('=============\\\\\\\=====================')
        idx = 0
        power_1 = 0
        for client in active_clients:
            print("train")
            # good_channel, power_1 =
            train(args, client, device)
            # if(good_channel == True):
            #     client_good_channel.append(client)
            # idx = idx+1

            # print(client)

    #     power = []

    #     for csi_i in csi :
    #         power.append(max(0,(1/mu - 1/csi_i)))
    #     # fig,ax=plt.subplots()
    #     # line1=ax.plot(csi,power,label="channel power allocated")
    #     # line2=ax.plot(csi,[1/mu_min]*len(csi),label="maximum power allocated")
    #     # ax.set_title("csi vs power allocated")
    #     # ax.set_xlabel("csi (channel gain to noise ratio)")
    #     # ax.set_ylabel("power allocated")
    #     # ax.legend()
    #     # plt.show()
    # #     # Testing
    # #     for client in active_clients:
    # #         test(args, client['model'], device, client['testset'], client['hook'].id)

    #     print()
    #     csi.sort()

    #     print("Clients with good channel are considered")
    #     for no in range (len(client_good_channel)):
    #         print(client_good_channel[no]['hook'].id)

    #     print()
    #     print("Sending data back to Server")
    #     print()

    #     #ClientUpdateVal(clients,key,key_array,power_client)
    #     #good_channel_odd,power_odd=ClientUpdateVal(client_good_channel,key,key_array,0)

    #     print()

    #     print("Clients having a good channel and considered for averaging")
    #     # for no in range (len(good_channel_odd)):
    #     #     print(good_channel_odd[no]['hook'].id)
        # Averaging
        print("jufhush")
        global_model = averageModels(global_model, active_clients)

        # Testing the average model
        test(args, global_model, device, global_test_loader, count)

        print("Total Power =", power_1)
        print()

        for client in clients:
            client['model'].load_state_dict(global_model.state_dict())

    if (args['save_model']):
        torch.save(global_model.state_dict(), "FederatedLearning.pt")

    print("============ Accuracy ===========")
    print(accu)
    return accu

#final_acc = []
#sum = []
#weight = []

# hook = sy.TorchHook(torch)

# for i in range(4):
#     accuracy1 = Wrapper(64,0.02,2,20,4,hook)
#     print(accuracy1)
#     if(len(sum)==0):
#         sum = accuracy1
#     for j in range(len(accuracy1)):
#         sum[j] = sum[j] + accuracy1[j]
#     # final_acc[i] = accuracy1
# for i in range(len(sum)):
#     sum[i] = sum[i]/10
#weight = sum/10

# # print(final_acc)


# print("====================final ans")
# # print(sum)
accuracy1 = Wrapper(64, 0.01, 3, 20, 10, key, key_array, Ps)
print(accuracy1)
# accuracy2 = Wrapper(64,0.02,2,20,5,hook)
# print(accuracy2)
# accuracy3 = Wrapper(64,0.02,2,20,5,hook)
# print(accuracy3)
