import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence

class Net(nn.Module):
    def __init__(self, num_embeddings=21, embedding_dim=32,
                 rnn_hidden_size=100, rnn_layers=2, rnn_dropout=0.3,
                 ffnn_out=20, ffnn_dropout=0.5,
                 batch_size=64, epochs=0):
        super(Net, self).__init__()

        self.num_epochs = epochs
        
        self.batch_size = batch_size
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        self.LSTM = nn.LSTM(embedding_dim, hidden_size=rnn_hidden_size, num_layers=rnn_layers,
                           batch_first=True, dropout=rnn_dropout, bidirectional=True)
        
        self.dropout = nn.Dropout(ffnn_dropout)
        
        self.FFNN = nn.Linear(2*rnn_hidden_size, ffnn_out, bias=True)

        self.activation = nn.Softmax(dim=1)
        
    def forward(self, batch, seq_lengths):
        
        embedded_batch = self.embedding(batch)
        
        packed = pack_padded_sequence(embedded_batch, seq_lengths, batch_first=True)
        
        rnn_out, (h_n, c_n) = self.LSTM(packed)
        x = rnn_out.data
        
        x = self.dropout(x)
        x = self.FFNN(x)
        
        return self.activation(x)
    