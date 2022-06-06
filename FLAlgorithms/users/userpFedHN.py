import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.trainmodel.models import *
from FLAlgorithms.users.userbase import User
import copy

# Implementation for pFeMe clients

class UserpFedHN(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate,beta,lamda,
                 local_epochs, optimizer, K, personal_learning_rate, n_kernels=16, hn_dataset=False):
        super().__init__(device, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, lamda,
                         local_epochs,hn_dataset=hn_dataset)


        
        data_name = 'cifar10'
        layer_config={'n_input': 100, 'n_output': 10 if data_name == 'cifar10' else 100}

        inner_lr = 5e-3
        inner_wd = 5e-5
        optimizer_config=dict(lr=inner_lr, momentum=.9, weight_decay=inner_wd)

        self.local_layers = LocalLayer(**layer_config).to(device)
        self.local_optimizer = torch.optim.SGD(self.local_layers.parameters(), **optimizer_config)

        self.personal_learning_rate = personal_learning_rate

        self.loss = nn.NLLLoss()
        self.eval_every = 100

        #in pfedHN we are not going to use this optimizer
        self.optimizer = torch.optim.SGD

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def test(self):
        self.model.eval()
        self.local_layers.eval()
        test_acc = 0
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            pred = self.local_layers(output)
            test_acc += (torch.sum(torch.argmax(pred, dim=1) == y)).item()
            #@loss += self.loss(output, y)
            #print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
            #print(self.id + ", Test Loss:", loss)
        return test_acc, y.shape[0]
     
    def test_acc_loss(self):
        self.model.eval()
        self.local_layers.eval()
        test_acc = 0
        loss = 0
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            pred = self.local_layers(output)
            test_acc += (torch.sum(torch.argmax(pred, dim=1) == y)).item()
            loss += self.loss(pred, y)
            #print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
            #print(self.id + ", Test Loss:", loss)
     
        return test_acc, loss, y.shape[0]


    def train_error_and_loss(self):
        self.model.eval()
        self.local_layers.eval()
        train_acc = 0
        loss = 0
        for x, y in self.trainloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            pred = self.local_layers(output)
            train_acc += (torch.sum(torch.argmax(pred, dim=1) == y)).item()
            loss += self.loss(pred, y)
            #print(self.id + ", Train Accuracy:", train_acc)
            #print(self.id + ", Train Loss:", loss)
        return train_acc, loss , self.train_samples