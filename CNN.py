# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 22:22:27 2021
@author: Jason
Classification model(CNN)
"""
'''import'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn
from sklearn.metrics import f1_score
'''superparameters'''
lr = 0.00005                # learning rate to train CNN
batchsize = 100             # batchsize to train CNN
num_epochs = 25             # total epochs of training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''CNN'''
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            # input:1*32*32;output:32*28*28
            nn.Conv2d(1, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # input:32*28*28;output:32*14*14
            nn.MaxPool2d(2, stride = 2)
        )
        self.layer2 = nn.Sequential(
            # input:32*14*14;output:64*12*12
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # input:64*12*12;output:64*6*6
            nn.MaxPool2d(2, stride = 2)
        )
        self.layer3 = nn.Sequential(
            # input:64*6*6;output:128*4*4
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),  
            # input:128*4*4;output:128*2*2
            nn.MaxPool2d(2, stride = 2)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.layer4(x)
        x = F.log_softmax(x,dim=1)
        return x
'''Training function of CNN'''
def Training(train_data, train_label, val_data, val_label, flag, dataflag):
    if flag==0:
        if dataflag==0:           
            name = 'model/MNIST/CNN.pt'
        else:
            name = 'model/Fashion-MNIST/CNN.pt'
    else:
        if dataflag==0:
            name = 'model/MNIST/CNN_reinforce.pt'
        else:
            name = 'model/Fashion-MNIST/CNN_reinforce.pt'
    # initialize parameters
    Net = CNN().to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(Net.parameters(), lr=lr)
    loss_all = []                 # prepare to draw loss curve on training set
    loss_treshhold = 9999         # loss treshhold
    count = 0
    for epoch in range(num_epochs):
        loss = 0                  # loss on training set in this epoch
        for i in range(int(train_data.size(0)/batchsize)):
            # train
            data = train_data[i*batchsize:(i+1)*batchsize].to(device)
            label = train_label[i*batchsize:(i+1)*batchsize].long().to(device)
            out = Net(data)
            theloss = loss_func(out, label)
            loss += theloss
            optimizer.zero_grad()
            theloss.backward()
            optimizer.step()
            print('Epoch[{}/{}] Step[{}/{}] loss:{:.8f}'.format(epoch+1, num_epochs, i+1, int(train_data.size(0)/batchsize), theloss.item()))
        loss = loss*1.0/(int(train_data.size(0)/batchsize))
        loss_all.append(loss.item())
        torch.save(Net, name)
        # # validation
        # val_out = Net(val_data.to(device))
        # loss_val = loss_func(val_out, val_label.long().to(device))
        # # if the loss is decreasing, save the model
        # #print(loss_val)
        # if loss_val < loss_treshhold:
        #     torch.save(Net, name)
        #     count = 0
        # # if the loss is increasing, the model may be overfitted
        # else:
        #     count += 1
        #     # the loss on validation set arise twice
        #     if count==2:
        #         print('The model seem to be overfitted!')
        #         break
        # loss_treshhold = loss_val   
'''Test the accuracy and F1score on the different training set'''
def Test(test_data, test_label, flag, dataflag):
    if flag==0:
        if dataflag==0:           
            name = 'model/MNIST/CNN.pt'
        else:
            name = 'model/Fashion-MNIST/CNN.pt'
    else:
        if dataflag==0:
            name = 'model/MNIST/CNN_reinforce.pt'
        else:
            name = 'model/Fashion-MNIST/CNN_reinforce.pt'
    # load network
    Net = torch.load(name)
    predict_label = torch.zeros(([test_data.shape[0]]))
    for i in range(int(test_data.shape[0]/batchsize)):
        data = test_data[i*batchsize:(i+1)*batchsize].to(device)
        predict = Net(data)
        predict_label[i*batchsize:(i+1)*batchsize] = predict.max(1, keepdim=True)[1].view(-1)
    # calculate accuracy and f1score
    f1 = f1_score(test_label, predict_label, average='micro')
    temp = predict_label - test_label
    accuracy = np.where(temp==0)[0].shape[0]/test_data.shape[0]
    return f1, accuracy
            
        
    
    
       
        



