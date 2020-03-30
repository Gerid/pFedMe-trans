import torch
import os
import numpy as np
import h5py
from utils.model_utils import Metrics
import copy

class Server:
    def __init__(self, dataset, model, batch_size, learning_rate,meta_learning_rate, lamda,
                 num_glob_iters, local_epochs, optimizer,num_users):

        # Set up the main attributes
        self.dataset = dataset
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        self.model = model
        self.users = []
        self.selected_users = []
        self.num_users = num_users
        self.meta_learning_rate = meta_learning_rate
        self.lamda = lamda
    
        # Initialize the server's grads to zeros
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
            param.grad = torch.zeros_like(param.data)
        # self.send_parameters()
        
    def aggregate_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)

    def add_grad(self, user, ratio):
        user_grad = user.get_grads()
        for idx, param in enumerate(self.model.parameters()):
            param.grad = param.grad + user_grad[idx].clone() * ratio

    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model)

    def add_parameters(self, user, ratio):
        model = self.model.parameters()
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        for user in self.select_users:
            self.add_parameters(user, user.train_samples / self.total_train_samples)

    def test(self):
        for user in self.users:
            user.test()

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))
    
    def select_users(self, round, num_users):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''
        if(num_users == len(self.users)):
            print("All users are selected")
            return self.users

        num_users = min(num_users, len(self.users))
        np.random.seed(round)
        return np.random.choice(self.users, num_users, replace=False) #, p=pk)

    # define function for persionalized agegatation.
    def persionalized_update_parameters(self, sum_model):
        for server_param,sum_params in zip(self.model.parameters(), sum_model):
            server_param.data = server_param.data - self.meta_learning_rate * (server_param.data- 1/self.num_users * sum_params.data)

    def sumall_parameters(self, sum_model, user):
        for sum_params, user_param in zip(sum_model, user.get_parameters()):
            sum_params.data = sum_params.data + user_param.data#.clone()
        return sum_model

    def persionalized_aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        sum_model = self.model.parameters()
        # Clear sum_model
        for param in sum_model:
            param.data = torch.zeros_like(param.data)

        for user in self.selected_users:
            self.sumall_parameters(sum_model,user)
        
        self.persionalized_update_parameters(sum_model)

    # Save loss, accurancy to h5 fiel
    def save(self):
        alg = self.dataset 

        alg = alg + "_" + str(self.learning_rate) + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b"
        with h5py.File("./results/"+'{}_{}.h5'.format(alg, self.local_epochs), 'w') as hf:
            hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
            hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
            hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
            hf.close()

    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        self.client_model.set_params(self.latest_model)
        for c in self.users:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, tot_correct

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]

        return ids, groups, num_samples, tot_correct, losses