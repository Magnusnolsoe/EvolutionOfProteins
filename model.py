import os
import torch.nn as nn
import torch.optim as optim

from data import DataLoader
from torch.nn import Softmax
from torch.nn.utils.rnn import pack_padded_sequence

class Net(nn.Module):
    def __init__(self, num_embeddings=21, embedding_dim=32,
                 rnn_hidden_size=100, rnn_layers=2, rnn_dropout=0.3,
                 ffnn_dropout=0.5, ffnn_out=20,
                 batch_size=64, epochs=0,
                 dataset_name=None, data_dir="data/"):
        super(Net, self).__init__()
        
        self.data_loader = DataLoader(os.path.join(data_dir, dataset_name), batch_size=batch_size, pad_sequences=True)
        self.num_epochs = epochs
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        self.RNN = nn.LSTM(embedding_dim, hidden_size=rnn_hidden_size, num_layers=rnn_layers,
                           batch_first=True, dropout=rnn_dropout, bidirectional=True)

        self.dropout = nn.Dropout(ffnn_dropout)
        
        self.FFNN = nn.Linear(2*rnn_hidden_size, ffnn_out)
        
        self.activation = Softmax(dim=1)
        
        self.optimizer = optim.SparseAdam()

    
    def forward(self, batch, seq_lengths):
        
        embedded = self.embedding(batch)
        
        packed = pack_padded_sequence(embedded, seq_lengths, batch_first=True)
        
        rnn_out, (h_n,c_n) = self.RNN(packed)

        x = self.dropout(rnn_out.data)
        x = self.FFNN(x) 
        
        return self.activation(x)
    
    def train(self):
        
        for epoch in range(self.num_epochs):
            batch_x, batch_seq_len, batch_target = next(self.data_loader)
            
            prediction = self.forward(batch_x, batch_seq_len)
            
            print(prediction)
            print(batch_target)