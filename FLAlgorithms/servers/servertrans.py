from re import A, S
from sklearn import model_selection
import torch
import logging
import os
import copy
import time
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

        self.recluster = False 
        # Initialize data for all  users
        if model[1] == 'dnn':
            emb_dim = 64
            self.attn_learning_rate = 0.05
            self.intra_attn_model = Attn_Model(emb_dim=64, attn_dim=64, num_heads=8).to(device)
            self.inter_attn_model = Attn_Model(emb_dim=64, attn_dim=64, num_heads=8).to(device)
        elif model[1] == 'cnn':
            emb_dim = 64 
            self.attn_learning_rate = 0.05
            self.intra_attn_model = Attn_Model().to(device)
            self.inter_attn_model = Attn_Model().to(device)
            self.recluster = True
        if model[1] != 'dnn':
            emb_dim = 128
            attn_dim = 128
            num_heads = 8 
            self.attn_learning_rate = 0.01
            self.intra_attn_model = Attn_Model().to(device)
            self.inter_attn_model = Attn_Model().to(device)
        data = read_data(dataset)
        total_users = len(data[0])
        self.K = K
        self.personal_learning_rate = personal_learning_rate
        self.prev_per_values = [0] * total_users

        self.net_values = [*self.model.state_dict().values()]
        self.per_values = self.net_values[-2:]
        self.emb_layer = nn.Linear(len(nn.utils.parameters_to_vector(self.per_values)),emb_dim).to(device)
        self.num_cluster = num_cluster

        self.logger = logging.getLogger('server')
        self.logger.setLevel(logging.DEBUG)
        #create file handler
        self.fh = logging.FileHandler('server.log')
        self.fh.setLevel(logging.INFO)

        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.fh.setFormatter(formatter)
        self.ch.setFormatter(formatter)
        self.logger.addHandler(self.fh)
        self.logger.addHandler(self.ch)
        
        self.logger.info('creating server')

        self.alpha_layer = nn.Linear(emb_dim, 1).to(device)
        self.attn_optimizer = torch.optim.SGD([
                {'params': self.emb_layer.parameters()},
                {'params': self.inter_attn_model.parameters()},
                {'params': self.intra_attn_model.parameters()},
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
        every_recluster_eps = 5 
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, "(Attn phase) -------------")

            print("Evaluate global model")
            print("")
            ep_start_time = time.time()
            self.evaluate()

            self.prev_per_values = []
            
            self.attn_optimizer.zero_grad()
            for i, user in enumerate(self.users):
                if glob_iter != 0:
                    user.prev_per_values = [0]*len(user.per_values)
                    self.copy_value(user.per_values, user.prev_per_values, if_grad=True)
                user.train(self.local_epochs)
                #get user embedding vec
                user.emb(self.emb_layer)
            
            #after local training, optimize attn modules, with loss betweeen prev_values and local updated
            if glob_iter != 0:
                self.attn_optimize()
            # self.cluster = []
            if glob_iter == 0:
                self.form_cluster()
                self.logger.info("iteration {} cluster results:".format(glob_iter))
                for idx, cluster in enumerate(self.clusters):
                    res = "cluster {} : ".format(idx)

                    for user in cluster.users:
                        res += str(user.id)
                        res += " "
                    self.logger.info(res)
                
            if self.recluster == True and glob_iter % every_recluster_eps == 0 and glob_iter != 0:
                self.form_cluster(self.recluster)
                self.logger.info("iteration {} cluster results:".format(glob_iter))
                for idx, cluster in enumerate(self.clusters):
                    res = "cluster {} : ".format(idx)

                    for user in cluster.users:
                        res += str(user.id)
                        res += " "
                    self.logger.info(res)

            for cluster in self.clusters:
                print("===================================")
                print("cluster users:")
                for user in cluster.users:
                    print(user.id)

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
                    user.per_values = self.weighted_agg_model([cluster.per_values, user.per_values], [1-alpha,alpha])
                    user.merge_base_per_model()
                cluster.merge_base_per_model()

            self.evaluate_personalized_model()
            ep_end_time = time.time()
            cost_time = ep_end_time - ep_start_time
            self.logger.info("iteration : {} totally cost time : {:.2f}".format(glob_iter, cost_time))

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
        user_emb_list = [user.emb_vec.clone().reshape(1, -1) for user in cluster.users]

        x = torch.cat(user_emb_list, dim=0).unsqueeze(1)
        if len(cluster.users) == 1:
            return
        weights = self.intra_attn_model(x).squeeze(0)
        user_model_list = [] 
        for i in range(len(cluster.users)):
            per_value = cluster.users[i].per_values
            for i, v in enumerate(per_value):
                per_value[i] = v.clone()
            user_model_list.append(per_value)
        weights = weights.squeeze(0)
        for i in range(weights.size()[0]):
            w = [weights[i][j] for j in range(weights[i].size()[0])]
            per_value = self.weighted_agg_model(user_model_list, w)
            cluster.users[i].per_values = per_value
            #user.per_values_temp = per_values


    def weighted_agg_model(self, models, weights, use_grad=False):
        res = []
        # note 'model' here is list of values ,
        for model, weight in zip(models, weights):
            if len(res) == 0:
                for value in model:
                    res.append(torch.zeros_like(value))
            for i in range(len(res)):
                if use_grad:
                    res[i] += model[i].clone() * weight
                else:
                    res[i] += model[i].data.clone() * weight
        return res



    def inter_cluster_agg(self):
        cluster_emb_list = [cluster.emb_vec.clone().reshape(1, -1) for cluster in self.clusters]
        x = torch.cat(cluster_emb_list, dim=0).unsqueeze(1)
        weights = self.inter_attn_model(x).squeeze(0)
        
        cluster_model_list = [cluster.per_values for cluster in self.clusters]
        for i in range(len(self.clusters)):
            per_value = self.clusters[i].per_values
            for i, v in enumerate(per_value):
                per_value[i] = v.clone()
            cluster_model_list.append(per_value)
        for i in range(weights.size()[0]):
            w = [weights[i][j] for j in range(weights[i].size()[0])]
            per_values = self.weighted_agg_model(cluster_model_list, w)
            self.clusters[i].per_values = per_values

    def attn_optimize(self):
        loss = 0
        total_train = 0
        self.attn_optimizer.zero_grad()
        for user in self.users:
            total_train += user.train_samples
        for i, user in enumerate(self.users):
            ratio = user.train_samples / total_train
            loss += ratio * torch.linalg.norm(nn.utils.parameters_to_vector(user.per_values) - nn.utils.parameters_to_vector(user.prev_per_values))
        loss.backward()
        self.attn_optimizer.step()

    def copy_value(self, value, value1=None,  if_grad=False, if_tensor=False):
        if value1 == None:
            value1 = [0] * len(value)
        assert (len(value1) == len(value))
        for i in range(len(value1)):
            if if_grad==False:
                value1[i] = value[i].detach()
            else:
                value1[i] = value[i].clone()
        if if_tensor == True:
            res = torch.tensor([]).to(self.device)
            for v in value1:
                res = torch.cat((res, v.unsqueeze(0)), 0)
            return res

        return value1

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        for i in range(len(self.clusters)):
            torch.save(self.clusters[i].model, os.path.join(model_path, "cluster",str(i) , ".pt"))

