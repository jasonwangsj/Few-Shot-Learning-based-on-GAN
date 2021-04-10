# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 22:23:10 2021
@author: Jason
DCGAN model
"""
'''import'''
import os
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import torchvision
from torchvision import utils, datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import pylab
import Augment
'''superparameters'''
batchsize = 100                     # batchsize for the training of DCGAN
classes = 10                        # the number of classes
noise_dim = 100                     # the dimension of noise
num_epochs = 600                    # train epochs
lr = 0.0002                         # learning rate
beta1 = 0.5                         # parameters used in Adam optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_dir = 'Img'                     # path which store real/fake image
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
# Label one-hot for G
# 1-hot vector for each digit
label_1hots = torch.zeros(10,10)
for i in range(10):
    label_1hots[i,i] = 1
label_1hots = label_1hots.view(10,10,1,1).to(device)
# Label one-hot for D
label_fills = torch.zeros(10, 10, 32, 32)
ones = torch.ones(32, 32)
for i in range(10):
    label_fills[i][i] = ones
label_fills = label_fills.to(device)
'''Generator'''
class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.image = nn.Sequential(
            # input:100*1*1;output:256*4*4
            nn.ConvTranspose2d(noise_dim, 64*4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64*4),
            nn.ReLU(True)
        )
        self.label = nn.Sequential(
            # input:10*1*1;output:256*4*4
            nn.ConvTranspose2d(classes, 64*4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64*4),
            nn.ReLU(True)
        )
        self.main = nn.Sequential(
            # input:512*4*4;output:256*8*8
            nn.ConvTranspose2d(64*8, 64*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*4),
            nn.ReLU(True),
            # input:256*8*8;output:128*16*16
            nn.ConvTranspose2d(64*4, 64*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*2),
            nn.ReLU(True),
            # input:128*16*16;output:1*32*32
            nn.ConvTranspose2d(64*2, 1, 4, 2, 1, bias=False),
            nn.Tanh(),  
        )
    def forward(self, image, label):
        image = self.image(image)
        label = self.label(label)
        out = torch.cat((image, label), 1)
        out = self.main(out)
        return out     
'''Discriminator'''
class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.image = nn.Sequential(
            Augment.AugmentPipe(),
            # input:1*32*32;output:64*16*16
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),         
            nn.LeakyReLU(0.2, True)
        )
        self.label = nn.Sequential(
            # input:10*32*32;output:64*16*16
            nn.Conv2d(classes, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True)
        )
        self.main = nn.Sequential(
            # input:128*16*16;output:256*8*8
            nn.Conv2d(64*2, 64*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*4),
            nn.LeakyReLU(0.2, inplace=True),
            # input:256*8*8;output:512*4*4
            nn.Conv2d(64*4, 64*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*8),
            nn.LeakyReLU(0.2, inplace=True),
            # input:512*4*4; output:1*1*1
            nn.Conv2d(64*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()    
        )
    def forward(self, image, label):
        image = self.image(image)
        label = self.label(label)
        out = torch.cat((image, label), dim=1)
        out = self.main(out)
        return out  
'''denormalization'''
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)
'''Parameter initialization'''
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
'''Training function of DCGAN'''
def Training(train_data, train_label, dataflag):
    # initialize
    NetG = G().to(device)
    NetD = D().to(device)
    NetG.apply(weights_init)
    NetD.apply(weights_init)
    loss_func = nn.BCELoss()
    G_optimizer = torch.optim.Adam(NetG.parameters(), lr=lr, betas=(beta1, 0.999))
    D_optimizer = torch.optim.Adam(NetD.parameters(), lr=lr, betas=(beta1, 0.999))
    G_treshhold = 100
    # begin to train
    for epoch in range(num_epochs):
        # for each epoch
        for i in range(int(train_data.size(0)/batchsize)):
            # data used to train
            real_img = train_data[i*batchsize:(i+1)*batchsize].to(device)
            real_label = torch.ones(([batchsize])).to(device)
            noise = torch.randn(batchsize, noise_dim, 1, 1).to(device)
            fake_label = torch.zeros(([batchsize])).to(device)
            label = train_label[i*batchsize:(i+1)*batchsize].long().to(device)
            G_label = label_1hots[label]
            D_label = label_fills[label]
            # first train the discriminator
            NetD.zero_grad()
            real_out = NetD(real_img, D_label).view(-1)
            real_score = real_out.data.mean()
            real_loss = loss_func(real_out, real_label)
            real_loss.backward()
            fake_img = NetG(noise, G_label)
            fake_out = NetD(fake_img, D_label).view(-1)
            fake_score = fake_out.data.mean()
            fake_loss = loss_func(fake_out, fake_label)
            fake_loss.backward()
            D_loss = real_loss + fake_loss
            D_optimizer.step()
            # train the generator
            NetG.zero_grad()
            noise = torch.randn(batchsize, noise_dim, 1, 1).to(device)
            fake_img = NetG(noise, G_label)
            out = NetD(fake_img, D_label).view(-1)
            score = out.data.mean()
            G_loss = loss_func(out, real_label)
            G_loss.backward()
            G_optimizer.step()
            # print information of training
            print(
                'Epoch:[{}/{}] step:[{}/{}] D_loss:{:.6f} G_loss:{:.6f} real_score:{:.6f} fake_score:{:.6f} score:{:.6f}'
                .format(epoch+1, num_epochs, i+1, int(train_data.size(0)/batchsize), D_loss.item(), G_loss.item(), real_score, fake_score, score)
            )
            # save the best model
            # if G_loss.item() < G_treshhold:
            #     G_treshhold = G_loss.item()
            #     torch.save(NetG.state_dict(), 'model/Generator.pt')
            if dataflag==0:
                torch.save(NetG.state_dict(), 'model/MNIST/Generator.pt')
            else:
                torch.save(NetG.state_dict(), 'model/Fashion-MNIST/Generator.pt')
        # save images for this epoch
        # save real image
        if epoch==0:
            real_img = train_data[0:batchsize]
            real_img = utils.make_grid(real_img, nrow=10)
            real_img = real_img.permute(1, 2, 0)*0.5+0.5
            plt.axis('off')
            plt.imshow(real_img)
            if dataflag==0:
                plt.savefig('Img/MNIST/real image.png')
            else:
                plt.savefig('Img/Fashion-MNIST/real image.png')
        # save fake image
        noise = torch.randn(batchsize, noise_dim, 1, 1).to(device)
        label = train_label[0:batchsize].long().to(device)
        G_label = label_1hots[label]
        with torch.no_grad():
            fake_img = NetG(noise, G_label)
        fake_img = utils.make_grid(fake_img, nrow=10)
        fake_img = fake_img.permute(1, 2, 0)*0.5+0.5
        plt.axis('off')
        plt.imshow(fake_img.cpu())
        if dataflag==0:
            plt.savefig('Img/MNIST/fake image epoch{}.png'.format(epoch+1))
        else:
            plt.savefig('Img/Fashion-MNIST/fake image epoch{}.png'.format(epoch+1))
'''Draw image of all digits'''   
def SaveDigits(dataflag):
    netG = G()
    if dataflag==0:        
        netG.load_state_dict(torch.load('model/MNIST/Generator.pt',map_location="cpu"))
    else:
        netG.load_state_dict(torch.load('model/Fashion-MNIST/Generator.pt',map_location="cpu"))
    netG.eval()
    fixed_noise = torch.randn(batchsize, noise_dim, 1, 1).to(device)
    fixed_label = label_1hots[torch.arange(10).repeat(10).sort().values]
    with torch.no_grad():
        fake = netG(fixed_noise.cpu(), fixed_label.cpu())
    fake_img = utils.make_grid(fake, nrow=10)
    fake_img = fake_img.permute(1, 2, 0)*0.5+0.5
    plt.axis('off')
    plt.imshow(fake_img.cpu())
    if dataflag==0:
        plt.savefig('Img/MNIST/fake image.png')
    else:
        plt.savefig('Img/Fashion-MNIST/fake image.png')
'''Use DCGAN to reinforce data'''
def DataReinforcement(train_data, train_label, dataflag):
    netG = G()
    if dataflag==0:
        netG.load_state_dict(torch.load('model/MNIST/Generator.pt', map_location=torch.device('cpu')))
    else:
        netG.load_state_dict(torch.load('model/Fashion-MNIST/Generator.pt', map_location=torch.device('cpu')))
    netG.eval()
    for digit in range(10): 
        print('-------------------------------------------')
        print('reinforce data with label {}………………'.format(digit))
        for i in range(60):
            print('{} data with label {} has been generated'.format(((i+1)*100), digit))
            fixed_noise = torch.randn(100, noise_dim, 1, 1)
            label = torch.tensor([digit]).repeat(100)
            fixed_label = label_1hots[label]
            with torch.no_grad():
                fake = netG(fixed_noise.cpu(), fixed_label.cpu())
            train_data = torch.cat((train_data, fake), 0)
            train_label = torch.cat((train_label, label.float()), 0)
    # shuffle
    index = [i for i in range(train_data.shape[0])]
    np.random.shuffle(index)
    train_data = train_data[index]
    train_label = train_label[index]
    return train_data, train_label
                