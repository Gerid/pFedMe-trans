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
        self.base_values = None
        self.per_values = None
        self.emb_vec = None
        self.selected_users = []

    
    def load_model(self, model):
        self.model.load_state_dict(model.state_dict())

    def update_model(self, emb_layer):
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
        self.base_values = self.net_values[:-2]
        value_vec = nn.utils.parameters_to_vector(self.per_values).clone()
        self.emb_vec = emb_layer(value_vec)
    

    def avg_update_model(self):
        total_train = 0
        for user in self.users:
            total_train += user.train_samples
        res_base_values = []
        res_per_values = []
        for user in self.users:
            ratio = user.train_samples / total_train
            if len(res_base_values) == 0 and len(res_per_values) == 0:
                if user.base_values != None:
                    for value in user.base_values:
                        res_base_values.append(torch.zeros_like(value))
                if user.per_values != None:
                    for value in user.per_values:
                        res_per_values.append(torch.zeros_like(value))
            if user.base_values != None:
                for v1, v2 in zip(res_base_values, user.base_values):
                    v1 += ratio * v2
            if user.per_values != None:
                for v1, v2 in zip(res_per_values, user.per_values):
                    v1 += ratio * v2
        self.base_values = res_base_values
        self.per_values = res_per_values


    def set_parameters(self, model):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()
        #self.local_weight_updated = copy.deepcopy(self.optimizer.param_groups[0]['params'])

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()
    
    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param
    
    def get_updated_parameters(self):
        return self.local_weight_updated
    
    def update_parameters(self, new_params):
        for param , new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def emb(self, emb_layer:nn.modules):
        value_vec = nn.utils.parameters_to_vector(self.per_values).clone()
        self.emb_vec = emb_layer(value_vec)






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
        self.net_values = [*self.model.state_dict().values()]
        self.per_values = self.net_values[-2:]
        self.base_values = self.net_values[:-2]
        self.emb_vec = None
        self.temp_per_values = None


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
        self.net_values = [*self.model.state_dict().values()]
        self.per_values = self.net_values[-2:]
        self.base_values = self.net_values[:-2]

        return LOSS
    
    def emb(self, emb_layer:nn.modules):
        self.net_values = [*self.model.state_dict().values()]
        self.per_values = self.net_values[-2:]
        value_vec = nn.utils.parameters_to_vector(self.per_values).clone()
        self.emb_vec = emb_layer(value_vec)
    
    def get_per_values_vec(self):
        return nn.utils.parameters_to_vector(self.per_values).clone()

    # using updated per_values and base_values to update self.model
    def merge_base_per_model(self):
        #res_model = copy.deepcopy(self.model)
        #for p1 ,p2 in zip(res_model.parameters(), self.model.parameters()):
            #p1.data = torch.zeros_like(p2.data)
            #p1.grad = torch.zeros_like(p2.data)
        for p in self.model.parameters():
            p.data = torch.zeros_like(p.data)
            p.grad = torch.zeros_like(p.data)

        res_values = []
        res_values.extend(self.base_values)
        res_values.extend(self.per_values)

        for v1, v2 in zip(self.model.state_dict().values, res_values):
            v1 += v2
        del res_values
