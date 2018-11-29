import torch

import torch.optim as optim
import torch.nn as nn

from model import Net
from data import DataLoader, DataIterator
from utils import custom_cross_entropy

def train(path="", epochs=1, batch_size=16):
    
    net = Net(embedding_dim=100)
    
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    criterion = custom_cross_entropy
    
    data_loader = DataLoader(path)
    
    X, y, seq = data_loader.run_pipline()
    
    train_iter = DataIterator(X[0], y[0], seq[0], batch_size=batch_size, pad_sequences=True)
    
    
    for epoch in range(epochs):
            
            for batch_x, batch_seq_len, batch_t in train_iter:
                
                predictions = net(batch_x, batch_seq_len)
                
                splitted = torch.split(predictions, batch_seq_len.tolist())
                
                b_size = batch_x.shape[0]
                
                batch_loss = criterion(splitted, batch_t, batch_seq_len)
                print(batch_loss)
                batch_loss.backward()
                optimizer.step()
                
                
                