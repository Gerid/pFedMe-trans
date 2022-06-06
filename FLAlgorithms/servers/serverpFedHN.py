from typing import OrderedDict
import torch
import random
import logging
import os
import time
from FLAlgorithms.users.userpFedHN import UserpFedHN
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
from FLAlgorithms.trainmodel.models import *
import numpy as np
from tqdm import trange


 
# Implementation for pFedHN Server

class pFedHN(Server):
    def __init__(self, device,  dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, K, personal_learning_rate, times, eval_every=50, embed_lr=None):
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
        
        self.eval_every = 1 
        self.device = device
        self.choice_per_iter = 6

        hyper_hid = 100
        n_hidden = 3
        n_kernels = 16
        self.hnet = CNNHyperPC(
            num_users, embed_dim, hidden_dim=hyper_hid, n_hidden=n_hidden,
            n_kernels=n_kernels
        )
        self.hnet = self.hnet.to(device)

        self.net = CNNTargetPC(n_kernels=n_kernels).to(device)
        self.net = self.net.to(device)
        lr = 5e-2
        embed_lr = embed_lr if embed_lr is not None else lr
        wd = 1e-3
        self.inner_wd = 5e-5
        self.inner_lr = 5e-3
        self.inner_steps = local_epochs

        optimizers = {
            'sgd': torch.optim.SGD(
                [
                    {'params': [p for n, p in self.hnet.named_parameters() if 'embed' not in n]},
                    {'params': [p for n, p in self.hnet.named_parameters() if 'embed' in n], 'lr': embed_lr}
                ], lr=lr, momentum=0.9, weight_decay=wd
            ),
            'adam': torch.optim.Adam(params=self.hnet.parameters(), lr=lr)
        }
        optim = "sgd"
        self.optimizer = optimizers[optim]
        criteria = torch.nn.CrossEntropyLoss()

        self.personal_learning_rate = personal_learning_rate
        for i in range(total_users):
            id, train , test = read_user_data(i, data, dataset)
            user = UserpFedHN(device, id, train, test, self.net, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, K, personal_learning_rate)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating pFedHN server.")

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
        step_iter = trange(self.num_glob_iters)
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " :pFedHN-------------")

            ep_start_time = time.time()

            self.hnet.train()
            for _ in range(self.choice_per_iter):
                node_id = random.choice(range(self.num_users))

                user = self.users[node_id]
                weights =self.hnet(torch.tensor([node_id], dtype=torch.long).to(self.device))
                self.net.load_state_dict(weights)
                user.set_parameters(self.net)

                #previous acc and loss
                prv_acc, prv_loss, ns = user.test_acc_loss()
                self.logger.info(
                    f"Step: {glob_iter+1}, User ID: {node_id}, Loss: {prv_loss:.4f},  Acc: {prv_acc/ns:.4f}"
                )
                # init inner optim
                inner_optimizer = torch.optim.SGD(
                    user.model.parameters(),lr=self.inner_lr,momentum=.9,
                    weight_decay=self.inner_wd
                )

                # storing theta_i for later calculating delta theta
                inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})

            
                for i in range(self.inner_steps):
                    user.model.train()
                    user.local_layers.train()
                    inner_optimizer.zero_grad()
                    self.optimizer.zero_grad()
                    user.local_optimizer.zero_grad()
                    #user.train(self.net)
                    X, y = user.get_next_train_batch()
                    inner_optimizer.step()
                    output = user.model(X)
                    pred = user.local_layers(output)
                    loss = user.loss(pred, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(user.model.parameters(), 50)
                    inner_optimizer.step()
                    user.local_optimizer.step()
            
                self.optimizer.zero_grad()
                final_state = user.model.state_dict()

                delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})

                # calculating phi gradient
                hnet_grads = torch.autograd.grad(
                    list(weights.values()), self.hnet.parameters(), grad_outputs=list(delta_theta.values())
                )

                # update hnet weights
                for p, g in zip(self.hnet.parameters(), hnet_grads):
                    p.grad = g

                torch.nn.utils.clip_grad_norm_(self.hnet.parameters(), 50)
                self.optimizer.step()

                
            if glob_iter % self.eval_every == 0:

                print("Evaluate global model")
                print("")
                last_eval = glob_iter
                self.hnet.eval()
                for node_id in range(self.num_users):
                    
                    emb_node = torch.tensor([node_id], dtype=torch.long)
                    emb_node = emb_node.to(self.device)
                    weights = self.hnet(emb_node)
                    self.users[node_id].model.load_state_dict(weights)
                self.evaluate()
                self.hnet.train()



            ep_end_time = time.time()
            cost_time = ep_end_time - ep_start_time
            print("ep cost time : {:.2f}".format(cost_time))


        #print(loss)
        self.save_results()

        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.hnet, os.path.join(model_path, "server.hyper_net" + ".pt"))
    
  
