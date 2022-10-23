import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, Dataset

#****************** ========== IID_Dataset ========== ******************** #


def mnistIID(data, nUsers):
    nImages = int(len(data)/nUsers)
    # print(len(data))
    # length of dataset is 60k
    usersDict, indices = {}, [i for i in (range(len(data)))]
    for i in range(nUsers):
        np.random.seed(i)
        usersDict[i] = set(np.random.choice(
            indices, nImages, replace=False))
        indices = list(set(indices)-usersDict[i])
        # print("i :::", end=" ")
        # print(usersDict)
        # print(len(usersDict), "-----------------",len(indices), "---000000----")
    return usersDict

    #************************ ======== Non-IID Dataset ========== ******************#


def mnistnon_IID(data, nuser):
    diff_class = 40
    images = int(len(data)/diff_class)
    diff_class_index = [i for i in range(diff_class)]
    usersDict = {i: np.array([]) for i in range(nuser)}
    indices = np.arange(diff_class*images)
    print(indices)
    unsorted_label = data.train_labels.numpy()
    #print(len(unsorted_label), "-----------")
    indices_unsorted = np.vstack((indices, unsorted_label))
    print("---*******")
    print(indices_unsorted)
    indices_label = indices_unsorted[:, indices_unsorted[1, :].argsort()]
    # print(indices_label, "*********")
    indices = indices_label[0, :]
    # print(indices, "0000")
    for i in range(nuser):
        # np.random.seed(i)
        print(diff_class_index, "-------")
        print(diff_class[i])
        temp = set(np.random.choice(diff_class_index, 2, replace=False))
        print(temp)
        diff_class_index = list(set(diff_class_index) - temp)
        for x in temp:
            usersDict[i] = np.concatenate(
                (usersDict[i], indices[x*images:(x+1)*images]), axis=0)
            # print(usersDict)
    return usersDict


class FedDataset(Dataset):
    def __init__(self, dataset, indx):
        self.dataset = dataset
        self.indx = [int(i) for i in indx]

    def __len__(self):
        return len(self.indx)

    def __getitem__(self, item):
        images, labels = self.dataset[self.indx[item]]
    #   print(type(torch.tensor(labels)))
        return ((images).clone().detach(), torch.tensor(labels))


def getImage(dataset, indices, batch_size):  # load images using the class FedDataset
    return DataLoader(FedDataset(dataset, indices), batch_size=batch_size, shuffle=True)
