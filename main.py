# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 20:04:19 2018

@author: magnu
"""

import os
import argparse
import torch
import torch.optim as optim
import pickle

from train import train
from utils import custom_cross_entropy, Logger
from model import Net

def main():


	parser = argparse.ArgumentParser()
	mutex_group = parser.add_mutually_exclusive_group()

	mutex_group.add_argument("-t", "--train", help="start the training", action="store_true")
	mutex_group.add_argument("-p", "--predict", help="make a prediction", action="store_true")

	# Data parameters
	parser.add_argument("--data_dir", help="relative path to the data directory (default is data/)", default="data")
	parser.add_argument("--dataset", help="filename of the dataset")

	# General parameters
	parser.add_argument("--gpu", help="specify whether to use gpu or not (default is false)", action="store_true")
	parser.add_argument("-v", "--verbose", help="specify the verbosity", action="store_true")
	parser.add_argument("--store_results", help="store training and test results", action="store_true")
	parser.add_argument("--checkpoint_dir", help="path to the checkpoint directory", default="checkpoint")
	parser.add_argument("--checkpoint_name", help="filename of checkpoint model", default="defualt.pt")
	parser.add_argument("--load_checkpoint", help="load checkpoint model", action="store_true")
	
	# Training parameters
	parser.add_argument("--epoch", help="number of epochs in training", type=int, default=1)
	parser.add_argument("--batch_size", help="size of batch in training", type=int, default=16)
	parser.add_argument("-lr", "--learning_rate", help="learning rate of optimizer", type=float, default=1e-4)
	parser.add_argument("--embedding_dim", help="dimension of embeddings", type=int, default=32)
	parser.add_argument("--rnn_layers", help="number of layers in RNN", type=int, default=2)
	parser.add_argument("--rnn_size", help="size of hidden units in RNN", type=int, default=100)
	parser.add_argument("--rnn_dropout", help="dropout rate in RNN", type=float, default=0.3)
	parser.add_argument("--dropout", help="dropout between RNN and linear layer", type=float, default=0.5)
	parser.add_argument("--linear_units", help="number of units in linear layer", type=int, default=20)
	parser.add_argument("--weight_decay", help="weight decay factor in optimizer (default is 0)", type=float, default=0)
	parser.add_argument("--split_rate", help="", type=float, default=0.25)
	
	args = parser.parse_args()
	
	logger = Logger(args.verbose)
	
	if args.train:
		if not args.dataset:
			raise Exception("Need to specify which dataset to train on!")

		data_path = os.path.join(args.data_dir, args.dataset)

		if not os.path.exists(data_path):
			raise Exception("{} path does not exist!".format(data_path))

		device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
		logger.info("Device in use: {}".format(device))
			
		if args.load_checkpoint:
			
			checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
			if not os.path.exists(checkpoint_path):
				raise Exception("{} path does not exist!".format(checkpoint_path))
			
			logger.info("Loading checkpoint")
			
			checkpoint = torch.load(checkpoint_path)
				
			net = Net(device).to(device)
			net.load_state_dict(checkpoint["model_state_dict"])
			
			optimizer = optim.Adam(net.parameters())
			optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
			
		else:

			net = Net(device, embedding_dim=args.embedding_dim,
			rnn_hidden_size=args.rnn_size, rnn_layers=args.rnn_layers, rnn_dropout=args.rnn_dropout,
			linear_out=args.linear_units, linear_dropout=args.dropout).to(device)
	
			optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
			
		criterion = custom_cross_entropy

		results = train(data_path, net, optimizer, criterion, device, logger, args)
		
		if args.store_results:
			if not os.path.exists("results"):
				os.mkdir("results")
			if not os.path.exists(args.checkpoint_dir):
				os.mkdir(args.checkpoint_dir)
				
			performance_name = "res.pk" # temporary name
			checkpoint_name = args.checkpoint_name # temporary name
			
			pickle.dump(results, open(os.path.join("results", performance_name), "wb"))
			torch.save({
					"epoch": args.epoch,
					"model_state_dict": net.cpu().state_dict(),
					"optimizer_state_dict": optimizer.state_dict() 
					}, os.path.join(args.checkpoint_dir, checkpoint_name))
	
if __name__ == "__main__":
	main()