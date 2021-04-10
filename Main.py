# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 22:24:13 2021
@author: Jason
Main function
"""
'''import'''
import Dataprocess
import DCGAN
import CNN
'''main function'''
def main():
    # change dataflag to choose the experimental data
    dataflag = 1
    # step1: Get training set and test set
    print('Downloading data………………')
    train_data, train_label, val_data, val_label, test_data, test_label = Dataprocess.GetData(dataflag)
    print('Downloading finished!')
    print('Begin to train DCGAN………………')
    # step2: Training CGAN and save the model
    DCGAN.Training(train_data, train_label, dataflag)
    DCGAN.SaveDigits(dataflag)
    print('Training finished!')
    # step3: Data reinforcement(using DCGAN)
    print('Begin to reinforce data with DCGAN………………')
    train_data_rein, train_label_rein = DCGAN.DataReinforcement(train_data, train_label, dataflag)
    print('Reinfocement finished!')
    # step4: Training CNN and save the model
    # train with few data
    print('Begin to train CNN with few data………………')
    CNN.Training(train_data, train_label, val_data, val_label, 0, dataflag)
    print('Training finished!')
    # train with reinforcement data
    print('Begin to train CNN with reinforcement data………………')
    CNN.Training(train_data_rein, train_label_rein, val_data, val_label, 1, dataflag)
    print('Training finished!')
    # step5: Performance comparation between the two models
    print('Begin to compare performance between the two model………………')
    f1, accuracy = CNN.Test(test_data, test_label, 0, dataflag)
    print('No reinforcement: Accuracy:{:.6f}  F1score{:.6f}'.format(accuracy, f1))
    f1, accuracy = CNN.Test(test_data, test_label, 1, dataflag)
    print('Reinforcement: Accuracy:{:.6f}  F1score{:.6f}'.format(accuracy, f1))    

if __name__ == '__main__':
    main()
    

