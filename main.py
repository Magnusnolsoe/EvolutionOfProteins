# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 20:04:19 2018

@author: magnu
"""

import os
import argparse

from model import Net
from data import DataLoader, DataIterator

def main():
    
    net = Net(epochs=1, embedding_dim=100, dataset_name="valid.txt")
    
    
    data_loader = DataLoader("valid.txt")
    data_loader.load_data()
    
    X_train, X_test, y_train, y_test, seq_train, seq_test = data_loader.split()
    
    X_train, y_train, seq_train = data_loader.sort_data(X_train, y_train, seq_train)
    
    train_iter = DataIterator(X_train, y_train, seq_train, batch_size=64)
    
    for epoch in range(1):
        
        batch_x, batch_seq_len, batch_t = next(train_iter)
        
        
        prediction = net(batch_x, batch_seq_len)
        '''
        print(prediction)
        print(prediction.shape)
        '''
        total = 0
        for t in batch_t:
            
            total += t.shape[0]
        print(total)
        print(prediction.shape[0])
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