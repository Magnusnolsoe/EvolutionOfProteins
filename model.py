import torch.nn as nn
import torch
from torch.autograd import Variable 

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Net(nn.Module):
    def __init__(self, device, num_embeddings=21, embedding_dim=32,
                 rnn_hidden_size=100, rnn_layers=2, rnn_dropout=0.3, bi_dir=True,
                 linear_out=20, linear_dropout=0.5):
        super(Net, self).__init__()
        
        self.device = device

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        self.rnn_hidden_size = rnn_hidden_size
        self.num_layers = rnn_layers
        self.directions= 2 if bi_dir else 1
        self.linear_units = linear_out
        
        self.fc1 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        
        self.LSTM = nn.LSTM(embedding_dim, hidden_size=rnn_hidden_size, num_layers=rnn_layers,
                           batch_first=True, dropout=rnn_dropout, bidirectional=bi_dir)
        
        self.dropout = nn.Dropout(linear_dropout)
        
        self.Linear = nn.Linear(self.directions*rnn_hidden_size, linear_out, bias=True)

        self.activation = nn.Softmax(dim=1)

    def init_hidden(self, batch_size):
        h0 = torch.randn(self.num_layers*self.directions, batch_size, self.rnn_hidden_size).to(self.device)
        c0 = torch.randn(self.num_layers*self.directions, batch_size, self.rnn_hidden_size).to(self.device)
        
        h0 = Variable(h0)
        c0 = Variable(c0)

        return (h0, c0)

        
    def forward(self, batch, seq_lengths):
        
        batch_size = len(batch)

        embedded_batch = self.embedding(batch)
        
        # Pack sequence
        packed = pack_padded_sequence(embedded_batch, seq_lengths, batch_first=True)
        
        hidden_state_params = self.init_hidden(batch_size)
        rnn_out, (h_n, c_n) = self.LSTM(packed, hidden_state_params)
        
        # Unpack sequence
        x, _ = pad_packed_sequence(rnn_out, batch_first=True, padding_value=0)
        
        max_seq_len = x.size()[1]
        
        x = x.contiguous()
        x = x.view(-1, x.shape[2])
    
        x = self.dropout(x)
        x = self.Linear(x)        
        x = self.activation(x)
        
        x = x.view(batch_size, max_seq_len, self.linear_units)
        
        return x
    