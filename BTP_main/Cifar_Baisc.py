# Name: Peng Cheng
# UIN: 674792652
from cProfile import run
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf

# Configuration area
#############################
epochs = 50
batch_size = 50
learning_rate = 0.001
num_workers = 4
load_pretrained_model = False
pretrained_epoch = 25
#############################
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainset_loader = torch.utils.data.DataLoader(cifar_trainset, batch_size=batch_size,shuffle=True, num_workers=num_workers)

cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testset_loader = torch.utils.data.DataLoader(cifar_testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, 5, padding=(1,1))
        self.conv2 = nn.Conv2d(48, 64, 5, padding=(1,1))
        self.conv3 = nn.Conv2d(64, 128, 5, padding=(1,1))
        # self.conv4 = nn.Conv2d(64, 256, 5, padding=(1,1))
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(128,256, 10, padding=(1,1))
        self.fc1 = nn.Linear(in_features=10*5*256, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.Dropout = nn.Dropout(0.15)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) #32*32*48
        x = F.relu(self.conv2(x)) #32*32*96
        x = self.pool(x) #16*16*96
        x = self.Dropout(x)
        x = F.relu(self.conv3(x)) #16*16*192
        # x = F.relu(self.conv4(x)) #16*16*256
        x = F.relu(self.conv4(x)) #16*16*256
        x = self.pool(x) # 8*8*256
        x = self.Dropout(x)
        x = x.view(-1, 10*5*256) # reshape x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.Dropout(x)
        x = self.fc3(x)
        return x

# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 48, 5, padding=(1,1))
#         self.conv2 = nn.Conv2d(48, 64, 5, padding=(1,1))
#         self.conv3 = nn.Conv2d(64, 128, 5, padding=(1,1))
#         self.conv4 = nn.Conv2d(128, 256, 5, padding=(1,1))
#         self.pool = nn.MaxPool2d(2,2)
#         self.conv5 = nn.Conv2d(256,50, 10, padding=(1,1))
#         self.fc1 = nn.Linear(in_features=5*5*50, out_features=512)
#         self.fc2 = nn.Linear(in_features=512, out_features=64)
#         self.Dropout = nn.Dropout(0.15)
#         self.fc3 = nn.Linear(in_features=64, out_features=10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x)) #32*32*48
#         x = F.relu(self.conv2(x)) #32*32*96
#         x = self.pool(x) #16*16*96
#         x = self.Dropout(x)
#         x = F.relu(self.conv3(x)) #16*16*192
#         x = F.relu(self.conv4(x)) #16*16*256
#         x = F.relu(self.conv5(x)) #16*16*256
#         x = self.pool(x) # 8*8*256
#         x = self.Dropout(x)
       
#         x = x.view(-1, 5*5*50) # reshape x
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.Dropout(x)
#         x = self.fc3(x)
#         return x
# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 48, 5, padding=(1,1))
#         self.conv2 = nn.Conv2d(48, 64, 5, padding=(1,1))
#         self.conv3 = nn.Conv2d(64, 128, 5, padding=(1,1))
#         self.conv4 = nn.Conv2d(128, 256, 5, padding=(1,1))
#         self.pool = nn.MaxPool2d(2,2)
#         self.fc1 = nn.Linear(in_features=5*5*256, out_features=512)
#         self.fc2 = nn.Linear(in_features=512, out_features=64)
#         self.Dropout = nn.Dropout(0.15)
#         self.fc3 = nn.Linear(in_features=64, out_features=10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x)) #32*32*48
#         x = F.relu(self.conv2(x)) #32*32*96
#         x = self.pool(x) #16*16*96
#         x = self.Dropout(x)
#         x = F.relu(self.conv3(x)) #16*16*192
#         x = F.relu(self.conv4(x)) #16*16*256
#         x = self.pool(x) # 8*8*256
#         x = self.Dropout(x)
#         x = x.view(-1, 5*5*256) # reshape x
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.Dropout(x)
#         x = self.fc3(x)
#         return x

# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, 5, padding=(1,1))
#         self.conv2 = nn.Conv2d(16, 48, 5, padding=(1,1))
#         self.conv3 = nn.Conv2d(48, 64, 5, padding=(1,1))
#         self.conv4 = nn.Conv2d(64, 256, 5, padding=(1,1))
#         self.pool = nn.MaxPool2d(2,2)
#         self.fc1 = nn.Linear(in_features=5*5*256, out_features=512)
#         self.fc2 = nn.Linear(in_features=512, out_features=64)
#         self.Dropout = nn.Dropout(0.22)
#         self.fc3 = nn.Linear(in_features=64, out_features=10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x)) #32*32*48
#         x = F.relu(self.conv2(x)) #32*32*96
#         x = self.pool(x) #16*16*96
#         x = self.Dropout(x)
#         x = F.relu(self.conv3(x)) #16*16*192
#         x = F.relu(self.conv4(x)) #16*16*256
#         x = self.pool(x) # 8*8*256
#         x = self.Dropout(x)
#         x = x.view(-1, 5*5*256) # reshape x
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.Dropout(x)
#         x = self.fc3(x)
#         return x

# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(3,48,3, padding=(1,1))
#         self.conv2 = nn.Conv2d(48,96,3, padding=(1,1))
#         self.conv3 = nn.Conv2d(96,192,3, padding=(1,1))
#         self.conv4 = nn.Conv2d(192,256,3, padding=(1,1))
#         self.pool = nn.MaxPool2d(2,2)
#         self.fc1 = nn.Linear(8*8*256, 512)
#         self.fc2 = nn.Linear(512, 64)
#         self.Dropout = nn.Dropout(0.25)
#         self.fc3 = nn.Linear(64,10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x)) #32*32*48
#         x = F.relu(self.conv2(x)) #32*32*96
#         x = self.pool(x) #16*16*96
#         x = self.Dropout(x)
#         x = F.relu(self.conv3(x)) #16*16*192
#         x = F.relu(self.conv4(x)) #16*16*256
#         x = self.pool(x) # 8*8*256
#         x = self.Dropout(x)
#         x = x.view(-1, 8*8*256) # reshape x
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.Dropout(x)
#         x = self.fc3(x)
#         return x

# Define loss function and optimizer. We employ cross-entropy and Adam
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
# define the test accuracy function

def test_accuracy(net, testset_loader, epoch):
    # Test the model
    net.eval()
    correct = 0
    total = 0 
    for data in testset_loader:
        images, labels = data
        images, labels = Variable(images).cuda(), labels.cuda()
        # print(labels)
        output = net(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        # correct = tf.get_static_value(correct)
        # correct = correct.numpy().tolist()
        index = correct.cpu().data.numpy().argmax()
    
    print(total)
    print(correct.data)
    vv = correct.cpu().clone().numpy()
    print(vv)
    v = tf.divide(vv,total)
    print('Accuracy of the network after epoch '+str(epoch+1)+' is: ' + str(100 * vv/total))
    
#We save the model after every 5 epochs
def save_model(net, epoch):
    filename = "model_with_epoch" + str(epoch+1) + ".pth"
    torch.save(net.state_dict(),filename)

#Train the neural network and test for accuracy after every 5 epochs
#Depend on whether we need to load the pretrained model
if load_pretrained_model == False:
    net = ConvNet()
    net.cuda()
    #optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    for epoch in range(epochs):
        running_loss = 0.0
        if epoch <= 10:
            optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        elif epoch > 10 and epoch <= 25:
            optimizer = optim.Adam(net.parameters(), lr=(learning_rate)/10)
        else:
            optimizer = optim.Adam(net.parameters(), lr=(learning_rate)/50)       
        for i, data in enumerate(trainset_loader):
            input_data, labels = data # data is a list of 2, the first element is 4*3*32*32 (4 images) the second element is a list of 4 (classes)
            input_data, labels = Variable(input_data).cuda(),Variable(labels).cuda()
            optimizer.zero_grad() # every time reset the parameter gradients to zero
            # forward backward optimize
            output = net(input_data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            # print the loss
            l = loss.data
            # print(loss.data)
            running_loss += loss.data
        # print the loss after every epoch
        tt = torch.div(running_loss,50000)
        print(running_loss)
        print('loss in epoch ' + str(epoch + 1) + ': ' + str(tt))    
        if (epoch + 1)%5 == 0:
            # Test for accuracy after every 5 epochs
            test_accuracy(net, testset_loader, epoch)
            # Save model after every 5 epochs
            save_model(net, epoch)
        elif epoch == epochs - 1:
            test_accuracy(net, testset_loader, epoch)
            save_model(net, epoch)
    

print("Training and Testing Completed!")