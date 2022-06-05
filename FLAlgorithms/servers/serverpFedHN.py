import torch
import logging
import os
import time
from FLAlgorithms.users.userpFedHN import UserpFedHN
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
from FLAlgorithms.trainmodel.models import *
import numpy as np

 
# Implementation for pFedMe Server

class pFedMe(Server):
    def __init__(self, device,  dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, K, personal_learning_rate, times ):
        super().__init__(device, dataset,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times)

        # Initialize data for all  users
        data = read_data(dataset)
        total_users = len(data[0])
        self.logger = logging.getLogger()
        self.K = K
        embed_dim = -1
        embed_dim = embed_dim
        if embed_dim == -1:
            embed_dim = int(1 + num_users / 4)

        hyper_hid = 100
        n_hidden = 3
        n_kernels = 16
        self.hnet = CNNHyperPC(
            num_users, embed_dim, hidden_dim=hyper_hid, n_hidden=n_hidden,
            n_kernels=n_kernels
        )

        self.personal_learning_rate = personal_learning_rate
        for i in range(total_users):
            id, train , test = read_user_data(i, data, dataset)
            user = UserpFedMe(device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, K, personal_learning_rate)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating pFedMe server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            # send all parameter for users 
            self.send_parameters()

            # Evaluate gloal model on user for each interation
            print("Evaluate global model")
            print("")
            ep_start_time = time.time()
            self.evaluate()

            # do update for all users not only selected users
            for user in self.users:
                user.train(self.local_epochs) #* user.train_samples
            
            # choose several users to send back upated model to server
            # self.personalized_evaluate()
            self.selected_users = self.select_users(glob_iter,self.num_users)

            # Evaluate gloal model on user for each interation
            #print("Evaluate persionalized model")
            #print("")
            self.evaluate_personalized_model()
            #self.aggregate_parameters()
            self.persionalized_aggregate_parameters()
            ep_end_time = time.time()
            cost_time = ep_end_time - ep_start_time
            print("ep cost time : {:.2f}".format(cost_time))


        #print(loss)
        self.save_results()
        self.save_model()
    
  
