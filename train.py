import torch

import torch.optim as optim
import torch.nn as nn

from model import Net
from data import DataLoader, DataIterator
from utils import target_to_tensor

def train():

    net = Net(epochs=1, embedding_dim=100)
    cuda = torch.cuda.is_available()

    if cuda:
        net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    
    data_loader = DataLoader("sample.txt")
    data_loader.load_data()
    
    X_train, X_test, y_train, y_test, seq_train, seq_test = data_loader.split()
    
    X_train, y_train, seq_train = data_loader.sort_data(X_train, y_train, seq_train)
    
    train_iter = DataIterator(X_train, y_train, seq_train, batch_size=16)
    
    
    for epoch in range(1):
            
            for batch_x, batch_seq_len, batch_t in train_iter:
                
                targets = target_to_tensor(batch_t)

                if cuda:
                    batch_x = batch_x.cuda()
                    targets = targets.cuda()

                predictions = net(batch_x, batch_seq_len)
                
                batch_loss = criterion(predictions, targets)

                batch_loss.backward()
                optimizer.step()

                print(batch_loss)