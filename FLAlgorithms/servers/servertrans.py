import torch
import os
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

        self.net_values = [*self.model.state_dict().values()]
        self.per_values = self.net_values[-2:]
        self.emb_layer = nn.Linear(len(nn.utils.parameters_to_vector(self.per_values)),128).to(device)
        self.num_cluster = num_cluster

        #intra_cluster_attn weight
        self.inter_query_weight = nn.Linear(emb_dim, attn_dim).to(device)
        self.inter_value_weight = nn.Linear(emb_dim, attn_dim).to(device)
        self.inter_LN = nn.LayerNorm(attn_dim).to(device)

        # 1-layer attention for simple verify
        self.inter_attn = nn.MultiheadAttention(attn_dim, num_heads).to(device)

        #inter_cluster_attn weight
        self.intra_query_weight = nn.Linear(emb_dim, attn_dim).to(device)
        self.intra_value_weight = nn.Linear(emb_dim, attn_dim).to(device)
        self.intra_LN = nn.LayerNorm(attn_dim).to(device)
    
        self.intra_attn = nn.MultiheadAttention(attn_dim, num_heads).to(device)

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

            self.send_parameters()
            print("Evaluate global model")
            print("")
            self.evaluate()

            for user in self.users:
                user.train(self.local_epochs)
                #get user embedding vec
                user.emb(self.emb_layer)
            
            # self.cluster = []
            if glob_iter % every_recluster_eps == 0:
                self.form_cluster()

            for cluster in self.clusters:
                self.intra_cluster_agg(cluster)

            self.inter_cluster_agg()
            

        #print(loss)
        self.save_results()
        self.save_model()
    
  
    def form_cluster(self, reform=False):
        #compute similarity k-means
        client_emb_list = [user.emb_vec.data.clone().reshape(1, -1) for user in self.users]
        client_emb_list = torch.cat(client_emb_list, dim=0)
        cluster_res = kmeans(X=client_emb_list, n_clusters=self.num_cluster, mode='euclidean')

        if reform:
            self.clusters = [Cluster(c_id, self.model) for c_id in range(self.num_cluster)]

        for client_id, cluster_id in enumerate(cluster_res):
           self.users[client_id].cluster_id = cluster_res[client_id] 
           self.clusters[cluster_id].users.append(self.users[client_id])
        
            #cluster.per_layer is the centroid per_model for clients within cluster
        
    def cluster_update(self):
        for cluster in self.clusters:
            cluster.update_model()
            cluster.emb_vec = self.emb_layer(cluster.per_values)

   
    def intra_cluster_agg(self, cluster):
        user_emb_list = [user.emb_vec.data.clone().reshape(1, -1) for user in cluster]

        x = torch.cat(user_emb_list, dim=0).unsqueeze(1)
        x = self.intra_LN(x) 
        q = self.intra_query_weight(x)
        k = self.intra_query_weight(x)
        v = torch.zeros_like(q)
        _, weights = self.intra_attn(q, k, v)
        user_model_list = [user.per_values for user in cluster.users]
        for w_i in weights:
            model = add_model(user_model_list, w_ij)
            user.
            


        user_emb_list = [user.emb_vec.data.clone().reshape(1, -1) for user in cluster]

    def weighted_agg_model(self, models, weights):
        res = None
        for model, weight in zip(models, weights):
            if res == None:
                res = copy.deepcopy(model)
                for key in res.state_dict().keys():
                    res.state_dict()[key] = torch.zeros_like(model.state_dict()[key])
            for param, new_param in zip(res.parameters(), model.parameters()):
                param += copy.deepcopy(new_param) * weight
        return res



    def inter_cluster_agg(self):
        pass
