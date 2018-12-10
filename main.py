# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 20:04:19 2018

@author: magnu
"""

import os
import argparse
import torch
import torch.optim as optim
import numpy as np

from pathlib import Path

from train import train
from utils import custom_cross_entropy
from model import Net
from data import DataLoader


def main():
        
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    
    group.add_argument("-t", "--train", help="start the training", action="store_true")
    group.add_argument("-p", "--predict", help="make a prediction", action="store_true")
    
    # Data parameters
    parser.add_argument("--data_dir", help="relative path to the data directory (default is data/)", default="data")
    parser.add_argument("--dataset", help="name of the dataset")
    parser.add_argument("--override", help="Override existing data in results folder", action="store_true")
    parser.add_argument("--output_dir", help="Directory for results", default="results")
    
    # General parameters
    parser.add_argument("--gpu", help="specify whether to use gpu or not (default is false)", action="store_true")
    parser.add_argument("-v", "--verbose", help="specify the verbosity", action="store_true")
    # parser.add_argument("--checkpoint_dir", help="path to the checkpoint directory", default="checkpoint")
    
    # Training parameters
    parser.add_argument("--epoch", help="number of epochs in training", type=int, default=1)
    parser.add_argument("--batch_size", help="size of batch in training", type=int, default=32)
    parser.add_argument("-lr", "--learning_rate", help="learning rate of optimizer", type=float, default=1e-4)
    parser.add_argument("--embedding_dim", help="dimension of embeddings", type=int, default=100)
    parser.add_argument("--rnn_layers", help="number of layers in RNN", type=int, default=2)
    parser.add_argument("--rnn_size", help="size of hidden units in RNN", type=int, default=512)
    parser.add_argument("--rnn_dropout", help="dropout rate in RNN", type=float, default=0.3)
    parser.add_argument("--dropout", help="dropout between RNN and linear layer", type=float, default=0.3)
    parser.add_argument("--linear_units", help="number of units in linear layer", type=int, default=20)
    parser.add_argument("--weight_decay", help="weight decay factor in optimizer (default is 0)", type=float, default=1e-5)
    parser.add_argument("--extra_dense_layer", help="Adds a dense layer between ebeddings and lstm", action="store_true")
    
    args = parser.parse_args()
    
    if args.train:
        if not args.dataset:
            raise Exception("Need to specify which dataset to train on!")
        
        data_path = os.path.join(args.data_dir, args.dataset)
        
        if not os.path.exists(data_path):
            raise Exception("{} path does not exist!".format(data_path))
        
        device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
        print("Device in use:", device)
       
        data_loader = DataLoader(data_path)
        split_rate = 0.20
        X, y, seq = data_loader.run_pipline(split_rate)

        ### Cross validation ###

        criterion = custom_cross_entropy

        for lr in np.logspace(-6, 0, 5):
            for weight_decay in np.logspace(-6, 0, 5):

                result_label = args.output_dir + "/lr=" + str(lr) + "-wd=" + str(weight_decay) + ".pk"
                result_path = Path(result_label)

                if not args.override and result_path.is_file():
                    print("Skipping training of model " + ', '.join(result_label.replace('.pk', '').split("/")[1].split('-')))
                else:
                    print("Training model " + ', '.join(result_label.replace('.pk', '').split("/")[1].split('-')))
                    net = Net(device=device,
                        embedding_dim=args.embedding_dim,
                        rnn_hidden_size=args.rnn_size,
                        rnn_layers=args.rnn_layers,
                        rnn_dropout=args.rnn_dropout,
                        linear_out=args.linear_units,
                        linear_dropout=args.dropout,
                        extra_dense_layer=args.extra_dense_layer).to(device)

                    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
                    train((X,y,seq), net, optimizer, criterion, device, args.epoch, args.batch_size, args.output_dir)

    
if __name__ == "__main__":
    main()