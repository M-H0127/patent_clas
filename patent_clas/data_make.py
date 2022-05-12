from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import torch
import numpy as np

def one_hot(label):
    max = np.max(label)
    label_list = torch.zeros(len(label), max+1)
    for i, j in enumerate(tqdm(label)):
        label_list[i,j] = 1
    return label_list

def te_tr_va_split(data, test_size, val_size, random_state_tr_te, random_state_tr_val, shuffle):
    val_size = val_size/(1-test_size)
    d_textlist = data["発明課題文"].values.tolist()
    a_textlist = data["要約"].values
    ipc_list = data["vector"].values.tolist()
    labellist = data["分類"].values.tolist()
    d_text_train_ten, d_text_test, a_text_train_ten, a_text_test, ipc_train_ten, ipc_test, label_train_ten, label_test = train_test_split(d_textlist, a_textlist, ipc_list, one_hot(labellist), test_size=test_size, shuffle=shuffle, random_state=random_state_tr_te)
    d_text_train, d_text_val, a_text_train, a_text_val, ipc_train, ipc_val, label_train, label_val = train_test_split(d_text_train_ten, a_text_train_ten, ipc_train_ten, label_train_ten, test_size=val_size, shuffle=shuffle, random_state=random_state_tr_val)
    return (d_text_test,a_text_test,ipc_test,label_test),(d_text_train,a_text_train,ipc_train,label_train),(d_text_val,a_text_val,ipc_val,label_val)

class createDataset(Dataset):
    def __init__(self, X, y, z, w):
        self.X = X
        self.y = y
        self.z = z
        self.w = w
    def __len__(self):
        return len(self.w)

    def __gettype__(self):
        return type(self.w)

    def __getitem__(self, index):
        a_text = self.X[index]
        d_text = self.y[index]
        ipc = self.z[index]
        labels = self.w[index]

        return a_text, d_text, ipc, labels
    
def createDataloader(dataset,batch_size):
    dataset = createDataset(*dataset)
    return DataLoader(dataset, batch_size=batch_size)

def main(data, test_size, val_size, random_state_tr_te=None, random_state_tr_val=None, shuffle=True, batch_size_te=32, batch_size_tr=32, batch_size_val=32):
    test, train, val = te_tr_va_split(data, test_size, val_size, random_state_tr_te, random_state_tr_val, shuffle)
    testloader = createDataloader(test, batch_size_te)
    trainloader = createDataloader(train, batch_size_tr)
    valloader = createDataloader(val, batch_size_val)
    return testloader, trainloader, valloader