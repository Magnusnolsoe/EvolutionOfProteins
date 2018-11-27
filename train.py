import torch

import torch.optim as optim
import torch.nn as nn

from model import Net
from data import DataLoader, DataIterator
from utils import target_to_tensor, pad_targets

def train(epochs=1, batch_size=16):
    
    net = Net(embedding_dim=100)
    
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    
    data_loader = DataLoader("testSample.txt")
    data_loader.load_data()
    
    X_train, X_test, y_train, y_test, seq_train, seq_test = data_loader.split()
    
    X_train, y_train, seq_train = data_loader.sort_data(X_train, y_train, seq_train)
    
    train_iter = DataIterator(X_train, y_train, seq_train, batch_size=batch_size, pad_sequences=True)
    
    
    for epoch in range(epochs):
            
            for batch_x, batch_seq_len, batch_t in train_iter:
                
                predictions = net(batch_x, batch_seq_len)
                
                splitted = torch.split(predictions, batch_seq_len.tolist())
                
                p1, _ = pad_targets(splitted, batch_seq_len)
                p, m = pad_targets(batch_t, batch_seq_len)
                
                print((p * torch.log(p1)).sum(2))
                print((p * torch.log(p1)).sum(2).shape)
                
                
                #batch_loss = criterion(predictions, targets)
                #batch_loss.backward()
                #optimizer.step()
                