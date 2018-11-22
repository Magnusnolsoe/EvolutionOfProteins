# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 20:04:19 2018

@author: magnu
"""

import os
import argparse
from model import Net

def main():
    
    network = Net(epochs=1, embedding_dim=100, batch_size=2, dataset_name="valid.txt")
    network.train()
    '''
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    
    group.add_argument("-t", "--train", help="start the training", action="store_true")
    group.add_argument("-p", "--predict", help="make a prediction", action="store_true")
    
    parser.add_argument("--data_dir", help="relative path to the data directory", default="data")
    parser.add_argument("--dataset", help="name of the dataset")
    parser.add_argument("--checkpoint_dir", help="path to the checkpoint directory", default="checkpoint")
    
    # Training parameters
    parser.add_argument("--epoch", help="number of epochs in training", type=int, default=3)
    
    args = parser.parse_args()
    
    if args.train:
        if not args.data_dir:
            raise Exception("Need to specify data directory when training!")
        if not os.path.exists(args.data_dir):
            raise Exception("{} directory does not exist!".format(args.data))
        
        
        # START TRAINING!
    '''
    
if __name__ == "__main__":
    main()