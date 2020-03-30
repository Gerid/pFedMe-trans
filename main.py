#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from fedl.servers.serveravg import FedAvg
from fedl.servers.serverfedl import FEDL
from fedl.servers.serverpsnl import Persionalized
from fedl.trainmodel.models import Mclr,Net

def main(dataset, algorithm, model, batch_size, learning_rate, meta_learning_rate, lamda, num_glob_iters,
         local_epochs, optimizer,numusers):

    if(model == "cnn"):
        model = Net(), model
    else:
        model = Mclr(),model

    if(algorithm == "FedAvg"):
        server = FedAvg(dataset, model, batch_size, learning_rate, meta_learning_rate, lamda, num_glob_iters, local_epochs, optimizer, numusers)
    
    if(algorithm == "Persionalized"):
        server = Persionalized(dataset, model, batch_size, learning_rate, meta_learning_rate, lamda, num_glob_iters, local_epochs, optimizer, numusers)

    server.train()
    server.test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Logistic_Synthetic",
                        choices=["Mnist", "Logistic_Synthetic"])
    parser.add_argument("--model", type=str, default="Mclr", choices=["cnn", "mclr"])
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate")
    parser.add_argument("--meta_learning_rate", type=float, default=0.01, help="Meta learning rate for global round")
    parser.add_argument("--lamda", type=float, default=0.01, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="Persionalized")
    parser.add_argument("--numusers", type=float, default=2, help="Number of Users per round") 
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Batch size: {}".format(args.batch_size))
    print("local learing rate       : {}".format(args.learning_rate))
    print("meta learing rate       : {}".format(args.meta_learning_rate))
    print("number user per round       : {}".format(args.numusers))
    print("K_g       : {}".format(args.num_global_iters))
    print("K_l       : {}".format(args.local_epochs))
    print("=" * 80)

    main(
        dataset=args.dataset,
        algorithm = args.algorithm,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        meta_learning_rate = args.meta_learning_rate, 
        lamda = args.lamda,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer= args.optimizer,
        numusers = args.numusers
    )