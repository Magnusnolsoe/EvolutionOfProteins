import os
import torch.nn as nn

from data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

class Net(nn.Module):
    def __init__(self, num_embeddings=21, embedding_dim=32,
                 rnn_hidden_size=100, rnn_layers=2, rnn_dropout=0.3,
                 batch_size=64, epochs=0,
                 dataset_name=None, data_dir="data/"):
        super(Net, self).__init__()
        
        self.data_loader = DataLoader(os.path.join(data_dir, dataset_name), batch_size=batch_size, pad_sequences=True)
        self.num_epochs = epochs
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        self.LSTM = nn.LSTM(embedding_dim, hidden_size=rnn_hidden_size, num_layers=rnn_layers,
                           batch_first=True, dropout=rnn_dropout, bidirectional=True)

    
    def forward(self, x):
        
        embedded = self.embedding(x)
        
        return embedded
    
    def train(self):
        
        for epoch in range(self.num_epochs):
            batch_x, batch_seq_len, batch_target = next(self.data_loader)
            
            emb = self.embedding(batch_x)
            
            packed = pack_padded_sequence(emb, batch_seq_len, batch_first=True)
            
            rnn_out, (h_n,c_n) = self.LSTM(packed)
            
            print(rnn_out.data.shape)
            
            
           #packed_x = pack_padded_sequence(batch_x, batch_seq_len, batch_first=True)
            
            '''
            for x, t in zip(batch_x, batch_target):   
                print(x)
                emb = self.forward(x)
                print(emb)
            '''