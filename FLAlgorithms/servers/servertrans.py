from re import A
import torch
import os
import copy

import copy
from kmeans_pytorch import kmeans
from torch import nn
from FLAlgorithms.users.userpFedMe import UserpFedMe
from FLAlgorithms.users.userpFedTrans import UserpFedTrans, Cluster
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np

from reformer_pytorch import *
 
# Implementation for pFedMe Server
class Attn_Model(nn.Module):
    def __init__(self, emb_dim=128, attn_dim=128, num_heads=8):
        super(Attn_Model, self).__init__()
        self.emb_dim = emb_dim
        self.attn_dim = attn_dim
        self.inter_query_weight = nn.Linear(emb_dim, attn_dim)
        self.inter_value_weight = nn.Linear(emb_dim, attn_dim)
        self.inter_LN = nn.LayerNorm(attn_dim)

        # 1-layer attention for simple verify
        self.inter_attn = nn.MultiheadAttention(attn_dim, num_heads)

    def forward(self, x, models=None, prev_models=None):
        x = self.inter_LN(x) 
        q = self.inter_query_weight(x)
        k = self.inter_query_weight(x)
        v = torch.zeros_like(q)

        _, weights = self.inter_attn(q, k, v)
        return weights

class pFedTrans(Server):
    def __init__(self, device,  dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, K, personal_learning_rate, times, num_cluster=10):
        super().__init__(device, dataset,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times)

        # Initialize data for all  users
        emb_dim = 128
        attn_dim = 128
        num_heads = 8 
        data = read_data(dataset)
        total_users = len(data[0])
        self.K = K
        self.personal_learning_rate = personal_learning_rate
        self.attn_learning_rate = 0.005
        self.prev_per_values = [0] * total_users

        self.net_values = [*self.model.state_dict().values()]
        self.per_values = self.net_values[-2:]
        self.emb_layer = nn.Linear(len(nn.utils.parameters_to_vector(self.per_values)),128).to(device)
        self.num_cluster = num_cluster

        #intra_cluster_attn weight
        self.attn_model = Attn_Model().to(device)

        # 1-layer attention for simple verify

        #inter_cluster_attn weight
        #self.intra_query_weight = nn.Linear(emb_dim, attn_dim).to(device)
        #self.intra_value_weight = nn.Linear(emb_dim, attn_dim).to(device)
        #self.intra_LN = nn.LayerNorm(attn_dim).to(device)
    
        #self.intra_attn = nn.MultiheadAttention(attn_dim, num_heads).to(device)
        self.alpha_layer = nn.Linear(emb_dim, 1).to(device)
        self.attn_optimizer = torch.optim.SGD([
                {'params': self.emb_layer.parameters()},
                {'params': self.attn_model.parameters()},
                {'params': self.alpha_layer.parameters()},
            ], lr=self.attn_learning_rate, momentum=0.9)
        self.attn_loss = nn.MSELoss().to(device)

        self.clusters = [Cluster(c_id, model[0]) for c_id in range(self.num_cluster)]
        for i in range(total_users):
            id, train , test = read_user_data(i, data, dataset)
            user = UserpFedTrans(device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, K, personal_learning_rate)
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
        every_recluster_eps = 5
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            # send all parameter for users 
            self.send_parameters()

            # Evaluate gloal model on user for each interation
            print("Evaluate global model")
            print("")
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
        
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, "(Attn phase) -------------")

            print("Evaluate global model")
            print("")
            self.evaluate()


            if glob_iter is not 0:
                self.attn_optimize()

            for user in self.users:
                user.prev_model = copy.deepcopy(user.model)
                user.train(self.local_epochs)
                #get user embedding vec
                user.emb(self.emb_layer)
            
            # self.cluster = []
            if glob_iter % every_recluster_eps == 0:
                self.form_cluster()

            #simply do FedAvg
            for cluster in self.clusters:
                cluster.avg_update_model()
                cluster.emb(self.emb_layer)

            for cluster in self.clusters:
                self.intra_cluster_agg(cluster)

            self.inter_cluster_agg()
            for cluster in self.clusters:
                for user in cluster.users:
                    alpha = self.alpha_layer(user.emb_vec)
                    alpha = torch.sigmoid(alpha)
                    user.per_values = self.model_add([cluster.per_values, user.per_values], [1-alpha,alpha])
                    user.merge_base_per_model()
                cluster.merge_base_per_model()

            self.attn_optimize()
            self.evaluate()

            for user in self.users:
                self.prev_per_values[user.id] = user.model


        #print(loss)
        self.save_results()
        self.save_model()
    
  
    def form_cluster(self, reform=False):
        #compute similarity k-means
        client_emb_list = [user.emb_vec.data.clone().reshape(1, -1) for user in self.users]
        client_emb_list = torch.cat(client_emb_list, dim=0)
        cluster_res = kmeans(X=client_emb_list, num_clusters=self.num_cluster, distance='euclidean', device=self.device)

        if reform:
            self.clusters = [Cluster(c_id, self.model) for c_id in range(self.num_cluster)]

        for client_id, cluster_id  in enumerate(cluster_res[0]):
           self.clusters[cluster_id].users.append(self.users[client_id])
        
            #cluster.per_layer is the centroid per_model for clients within cluster
        
    def model_add(self, models, weights):
        res = copy.deepcopy(models[0])
        for param in res.parameters():
            param.data = torch.zeros_like(param.data)
            param.grad = torch.zeros_like(param.data)
            print(param)
        for model, weight in zip(models, weights):
            for res_param, model_param in zip(res.parameters(), model.parameters()):
                res_param.data += model_param.data.clone() * weight
        return res
    
    def cluster_update(self):
        for cluster in self.clusters:
            cluster.update_model()
            cluster.emb_vec = self.emb_layer(cluster.per_values)

   
    def intra_cluster_agg(self, cluster):
        user_emb_list = [user.emb_vec.data.clone().reshape(1, -1) for user in cluster.users]

        x = torch.cat(user_emb_list, dim=0).unsqueeze(1)
        weights = self.attn_model(x).squeeze(0)
        user_model_list = [copy.deepcopy(user.per_values) for user in cluster.users]
        weights = weights.squeeze(0)
        print('weights.size:', weights.size())
        for i in range(weights.size()[0]):
            print('weights[i].size', weights[i].size())
            w = [weights[i][j] for j in range(weights[i].size()[0])]
            cluster.users[i].per_values = self.weighted_agg_model(user_model_list, w)
            #user.per_values_temp = per_values


    def weighted_agg_model(self, models, weights):
        res = []
        # note 'model' here is list of values ,
        for model, weight in zip(models, weights):
            if len(res) == 0:
                for value in model:
                    res.append(torch.zeros_like(value))
            for i in range(len(res)):
                res[i] += copy.deepcopy(model[i]) * weight
        return res



    def inter_cluster_agg(self):
        cluster_emb_list = [cluster.emb_vec.data.clone().reshape(1, -1) for cluster in self.clusters]
        x = torch.cat(cluster_emb_list, dim=0).unsqueeze(1)
        weights = self.attn_model(x)
        
        cluster_model_list = [cluster.per_values for cluster in self.clusters]
        for w_i, cluster in zip(weights,self.clusters):
            per_values = self.weighted_agg_model(cluster_model_list, w_i)
            cluster.per_values = per_values


    def attn_optimize(self):
        loss = 0
        self.attn_optimizer.zero_grad()
        for user in self.users:
            total_train += user.train_samples
        for user in self.users:
            ratio = user.train_samples / total_train
            loss += ratio*torch.linalg.norm(user.per_values - self.prev_per_values[user.id])
        loss.backward()
        self.attn_optimizer.step()
