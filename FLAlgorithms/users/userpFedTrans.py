from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.optimizers.fedoptimizer import pFedMeOptimizer
from FLAlgorithms.users.userbase import User
import copy

# Implementation for pFeMe clients

class Cluster():
    def __init__(self, cluster_id, model ):
        self.cluster_id = cluster_id
        self.users = []
        self.model = copy.deepcopy(model)
        self.net_values = None
        self.per_values = None
        self.selected_users = []

    
    def load_model(self, model):
        self.model.load_state_dict(model.state_dict())

    def update_model(self):
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        for user in self.users:
            total_train += user.train_samples
        for user in self.users:
            ratio = user.train_samples / total_train
            for cluster_param, user_param in zip(self.model.parameters(), user.get_parameters()):
                cluster_param.data = cluster_param.data + user_param.data.clone() * ratio

        self.net_values = [*self.model.state_dict().values()]
        self.per_values = self.net_values[-2:]




class UserpFedTrans(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate,beta,lamda,
                 local_epochs, optimizer, K, personal_learning_rate):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, beta, lamda,
                         local_epochs)

        if(model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()

        self.K = K
        self.personal_learning_rate = personal_learning_rate
        self.optimizer = pFedMeOptimizer(self.model.parameters(), lr=self.personal_learning_rate, lamda=self.lamda)
        
        self.cluster_id = None
        self.emb_vec = None


    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs):
        LOSS = 0
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):  # local update
            
            self.model.train()
            X, y = self.get_next_train_batch()

            # K = 30 # K is number of personalized steps
            for i in range(self.K):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.persionalized_model_bar, _ = self.optimizer.step(self.local_model)

            # update local weight after finding aproximate theta
            for new_param, localweight in zip(self.persionalized_model_bar, self.local_model):
                localweight.data = localweight.data - self.lamda* self.learning_rate * (localweight.data - new_param.data)

        #update local model as local_weight_upated
        #self.clone_model_paramenter(self.local_weight_updated, self.local_model)
        self.update_parameters(self.local_model)

        return LOSS
    
    def emb(self, emb_layer:nn.modules):
        self.net_values = [*self.model.state_dict().values()]
        self.per_values = self.net_values[-1:]
        self.emb_vec = emb_layer(self.per_values)
