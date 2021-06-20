import collections

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


## Function for loading the postive data points
def loadPositive():
    batch_size = 100
    with open('positive.txt','r',encoding="utf-8") as f:
        sents = [x for x in f.read().split('\n')]
    return DataLoader(sents, batch_size, shuffle=False)


## Function for loading the negative data points
def loadNegative():
    batch_size = 100
    with open('negative.txt','r',encoding="utf-8") as f:
        sents = [x for x in f.read().split('\n')]
    return DataLoader(sents, batch_size, shuffle=False)


## Function for loading the both positve and negative data points
def load():
    batch_size = 100
    label = []
    data = []
    with open('data.txt','r',encoding="utf-8") as f:
        sents = [x for x in f.read().split('\n')]
        for i in sents:
            if(len(i) > 0):
                temp = i.split(",")
                label.append([float(temp[-1])])
                temp = temp[0:-1]
                sent = " ".join(temp)
                data.append(sent)
    return  DataLoader(data,  batch_size, shuffle=False)


## Function for loading the lables of datapoints of load() function 
def getlabel():
    batch_size = 100
    label = []
    data = []
    with open('data.txt','r',encoding="utf-8") as f:
        sents = [x for x in f.read().split('\n')]
        for i in sents:
            if (len(i) > 0):
                temp = i.split(",")
                label.append([float(temp[-1])])
                temp = temp[0:-1]
                sent = " ".join(temp)
                data.append(sent)
        label = np.array(label)
        label = np.where(label < 0, 1, 0)
        label = torch.from_numpy(label)
        label = label.type(torch.float)
    return   DataLoader(label, batch_size, shuffle=False)
