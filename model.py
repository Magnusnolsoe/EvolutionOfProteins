import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence

class Net(nn.Module):
    def __init__(self, num_embeddings=21, embedding_dim=32,
                 rnn_hidden_size=100, rnn_layers=2, rnn_dropout=0.3, bi_dir=True,
                 linear_out=20, linear_dropout=0.5):
        super(Net, self).__init__()
        
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        self.rnn_hidden_size = rnn_hidden_size
        self.num_layers = rnn_layers
        self.directions= 2 if bi_dir else 1
        rnn_dropout = rnn_dropout if rnn_layers > 1 else 0
        
        self.LSTM = nn.LSTM(embedding_dim, hidden_size=rnn_hidden_size, num_layers=rnn_layers,
                           batch_first=True, dropout=rnn_dropout, bidirectional=bi_dir)
        
        self.dropout = nn.Dropout(linear_dropout)
        
        self.Linear = nn.Linear(self.directions*rnn_hidden_size, linear_out, bias=True)

        self.activation = nn.Softmax(dim=1)
        
    def forward(self, batch, seq_lengths):
        
        embedded_batch = self.embedding(batch)
        
        packed = pack_padded_sequence(embedded_batch, seq_lengths, batch_first=True)
        
        rnn_out, (h_n, c_n) = self.LSTM(packed)
        x = rnn_out.data

        x = self.dropout(x)
        x = self.Linear(x)
        
        return self.activation(x)
    