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
from functions import mnistIID, mnistnon_IID, FedDataset, getImage
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
    pz = 0
    print("yes")
    args = {
        'batch_size': 64,
        'test_batch_size': 1000,
        'lr': 0.005,
        'log_interval': 10,
        'epochs': 2,
        'clients': 20,
        'seed': 0,
        'rounds': 150,
        'C': 0.9,
        'lowest_snr': 20,
        'highest_snr': 30,
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
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

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
        client['flag'] = True

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

    def train(args, client, device, csi, snr, mu, key, key_array):
        cStatus = False
        client['model'].train()
        # snr = random.randint(0, 40)

        if(client['flag'] == True):
            if(csi == 0 or mu == 0):
                Optimal_Power = 0
            else:
                Optimal_Power = max(0, (1/mu - 1/csi))
            client['Optimal_Power'] = Optimal_Power
            # Optimal_Power = max(0,(1/mu - 1/csi))
            client['flag'] = False

        print("Client:", client['hook'].id)
        print()
        Optimal_Power = client['Optimal_Power']
        print("Optimal power allocated is: ", Optimal_Power)
        print("SNR==", snr)
        print("CSI ==", csi)

        snr_val = 10**(snr/10)
        absOfH = csi*Optimal_Power/snr_val
        x = random.uniform(0, absOfH)
        y = math.sqrt(absOfH*absOfH-x*x)
        #std = math.sqrt(Ps/snr_val)
        std = math.sqrt(absOfH*absOfH - x*x)
        h = complex(x, y)
        if(Optimal_Power != 0):
            data = client['model'].conv1.weight
            data = data*math.sqrt(Optimal_Power)
            noise = torch.randn(data.size())
            y_out = h*data + noise*std
            y_out = y_out/(math.sqrt(Optimal_Power)*(h))
            y_out = y_out.real

            client['model'].conv1.weight.data = y_out

            y_out = client['model'].conv2.weight
            y_out = y_out*math.sqrt(Optimal_Power)
            noise = torch.randn(y_out.size())
            y_out = h*y_out + noise*std
            y_out = y_out/(math.sqrt(Optimal_Power)*(h))
            y_out = y_out.real

            client['model'].conv2.weight.data = y_out

        client['model'].send(client['hook'])
        print("Client:", client['hook'].id)
        # print("CSI", abs(h)/(std*std))

        key_received = h*key_array+(np.random.randn(len(key_array))*std*2)
        # print(key_array_received)
        key_received = (key_received/(h)).real

        for n in range(len(key_received)):
            if(key_received[n] >= 0):
                key_received[n] = 0
            else:
                key_received[n] = 1

        key_received = key_received.tolist()
        key_received = [int(item) for item in key_received]
        # # iterate over federated data

        Xor_sum = sum(np.bitwise_xor(key, key_received))
        error = Xor_sum/len(key)
        # error = 0
        # if(error == 0 and Optimal_Power >0):
        if(round(error) == 0 and Optimal_Power > 0):
            cStatus = True     # Client status
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

                    # print("==========ye chalega kya========================")
                    if batch_idx % args['log_interval'] == 0:
                        loss = loss.get()
                        # print(loss.item())
                        print('Model {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            client['hook'].id,
                            epoch, batch_idx *
                            args['batch_size'], len(
                                client['mnist_trainset']) * args['batch_size'],
                            100. * batch_idx / len(client['mnist_trainset']), loss.item()))
        else:
            print("Channel is not taken for fedavg in this round")
        client['model'].get()

        # return cStatus

        if(Optimal_Power != 0):
            y_out = client['model'].conv1.weight
            y_out = y_out*math.sqrt(Optimal_Power)
            y_out = h*y_out+(torch.randn(y_out.size())*std)
            y_out = y_out/(math.sqrt(Optimal_Power)*(h))
            y_out = y_out.real
            client['model'].conv1.weight.data = y_out

            y_out = client['model'].conv2.weight
            y_out = y_out*math.sqrt(Optimal_Power)
            y_out = h*y_out + (torch.randn(y_out.size())*std)
            y_out = y_out/(math.sqrt(Optimal_Power)*(h))
            y_out = y_out.real
            client['model'].conv2.weight.data = y_out

            print()
        return cStatus

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

    rc = 1

    for fed_round in range(args['rounds']):
        # print("fed_round")
        # number of selected clients
        client_good_channel = []

        # at least 1 client is selected for train
        m = int(max(args['C'] * args['clients'], 1))

        # # dont choose same client more than once
        selected_clients_inds = np.random.choice(
            range(len(clients)), m, replace=False)
        selected_clients = [clients[i] for i in selected_clients_inds]

        # Active devices
        np.random.seed(fed_round)
        active_clients_inds = np.random.choice(selected_clients_inds, int(
            (1-args['drop_rate']) * m), replace=False)  # drop clients
        active_clients = [clients[i] for i in active_clients_inds]

        idx = 0
        power_1 = []
        print("**********************")
        print(pz)
        print("**********************")
        if(pz == 0):
            csi = []
            snr = []

            for ii in range(int(args['clients'])):
                csi.append(random.uniform(
                    args['lowest_csi'], args['highest_csi']))
                snr.append(random.randint(
                    args['lowest_snr'], args['highest_snr']))

            smallmu1 = 0
            gsmall1 = 3.402823466E+38

            mu = 1e-15
            while(mu <= 1):
                g1 = 0
                pn1 = 0

                for j in csi:
                    pn = max(1/mu-1/j, 0)
                    g1 += math.log(1+pn*j)
                    pn1 += pn
                g = g1-mu*(pn1-Ps*(int(args['clients'])-1))
                if(g < gsmall1):
                    smallmu1 = mu
                    gsmall1 = g
                mu += 0.00002
            pz = 10

        print(fed_round)
        for client in clients:
            print("train")
            good_channel = train(args, client, device,
                                 csi[idx], snr[idx], smallmu1, key, key_array)

            if(good_channel == True):
                client_good_channel.append(client)
            idx = idx+1

            # print(client)'
        print()
        # for csx in csi:
        #     power_1.append(max((1/smallmu1-1/csx), 0))

        # plt.bar([str(i) for i in range(1, len(power_1)+1)], power_1,)
        # csi.sort()
        # po = []
        # for jj in csi:
        #     po.append(max(1/smallmu1-1/jj, 0))
        # fig, ax = plt.subplots()
        # line1 = ax.plot(csi, po, label="channel power allocated")
        # line2 = ax.plot(csi, [1/smallmu1]*len(csi),
        #                 label="maximum power allocated")
        # ax.set_title("csi vs power allocated")
        # ax.set_xlabel("csi (channel gain to noise ratio)")
        # ax.set_ylabel("power allocated")
        # ax.legend()

        print("reached this step")

        print("Clients with good channel are considered for averaging")
        for no in range(len(client_good_channel)):
            print(client_good_channel[no]['hook'].id)
        print()

        global_model = averageModels(global_model, client_good_channel)

        # Testing the average model
        test(args, global_model, device, global_test_loader, count)

        #print("Total Power =", power_1)
        print()

        for client in clients:
            client['model'].load_state_dict(global_model.state_dict())

    if (args['save_model']):
        torch.save(global_model.state_dict(), "FederatedLearning.pt")

    print("============ Accuracy ===========")
    # print(accu)
    return accu


# print("====================final ans")
# # print(sum)

accuracy1 = Wrapper(64, 0.01, 3, 20, 10, key, key_array, Ps)
print(accuracy1)
# accuracy2 = Wrapper(64,0.02,2,20,5,hook)
# print(accuracy2)
# accuracy3 = Wrapper(64,0.02,2,20,5,hook)
# print(accuracy3)
