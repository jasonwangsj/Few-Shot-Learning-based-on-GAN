# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 22:23:40 2021
@author: Jason
Data load and process function
"""
'''import'''
import torch
import torchvision
from torchvision import transforms, datasets, models
'''superparameters'''
batch_size = 32
'''download MNIST data and divide it into training set and test set'''
def GetData(dataflag):
    # convert the image into tensor and normalization
    dataTransform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5,
                             std=0.5)    
    ])
    if dataflag==0:          # Using MNIST   
        # download MNIST dataset
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                'data', 
                download = False, 
                train = True,
                transform = dataTransform
            ), 
            batch_size = batch_size, 
            shuffle = True
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                'data',
                download = False,
                train = False,
                transform = dataTransform
            ),
            batch_size = batch_size,
            shuffle = True
        )  
    else:                     # Using Fashion-MNIST
        # download FashionMNIST dataset
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                'data', 
                download = False, 
                train = True,
                transform = dataTransform
            ), 
            batch_size = batch_size, 
            shuffle = True
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                'data',
                download = False,
                train = False,
                transform = dataTransform
            ),
            batch_size = batch_size,
            shuffle = True
        )  
    # create training set(few shot) and test set
    # training set contains 320 images
    train_data = torch.empty((800,1,32,32))
    train_label = torch.empty(([800]))
    for i, (data, label) in enumerate(train_loader):
        train_data[i*batch_size:(i+1)*batch_size] = data
        train_label[i*batch_size:(i+1)*batch_size] = label
        if i==24:
            break
    val_data = torch.empty((800,1,32,32))
    val_label = torch.empty(([800]))
    for j, (data, label) in enumerate(train_loader):
        if j>=100:
            i = j-100
            val_data[i*batch_size:(i+1)*batch_size] = data
            val_label[i*batch_size:(i+1)*batch_size] = label
        if j==124:
            break
    # test set contains 10000 images
    test_data = torch.empty((10000,1,32,32))
    test_label = torch.empty(([10000]))    
    for i, (data, label) in enumerate(test_loader):
        test_data[i*batch_size:(i+1)*batch_size] = data
        test_label[i*batch_size:(i+1)*batch_size] = label
    return train_data, train_label, val_data, val_label, test_data, test_label
    