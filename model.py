import torch.nn as nn
import torch
import crf; import sys

from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import relu, sigmoid

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
        rnn_dropout = rnn_dropout if rnn_layers > 1 else 0
        
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
        
        n = x.size()[1]
        
        x = x.contiguous()
        x = x.view(-1, x.shape[2])
    
        x = self.dropout(x)
        x = self.Linear(x)        
        x = self.activation(x)
        
        x = x.view(batch_size, n, self.linear_units)
        
        return x


class CRF_Net(nn.Module):
    def __init__(self, device, num_embeddings=21, embedding_dim=32,
                 rnn_hidden_size=100, rnn_layers=2, rnn_dropout=0.3, bi_dir=True,
                 linear_units=200, classes=20, linear_dropout=0.5):
        super(CRF_Net, self).__init__()
        
        self.device = device

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        self.rnn_hidden_size = rnn_hidden_size
        self.num_layers = rnn_layers
        self.directions= 2 if bi_dir else 1
        self.classes = classes
        self.linear_units = linear_units
        rnn_dropout = rnn_dropout if rnn_layers > 1 else 0
        
        self.LSTM = nn.LSTM(embedding_dim, hidden_size=rnn_hidden_size, num_layers=rnn_layers,
                           batch_first=True, dropout=rnn_dropout, bidirectional=bi_dir)
        
        self.dropout = nn.Dropout(linear_dropout)
        
        self.linear = nn.Linear(self.directions*rnn_hidden_size, linear_units)
        
        self.f = nn.Linear(linear_units, classes, bias=True)
        self.g = nn.Linear(2*linear_units, classes**2, bias=True)


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
        h, _ = pad_packed_sequence(rnn_out, batch_first=True, padding_value=0)
        n = h.size()[1]
        
        h = h.contiguous().view(-1, h.size()[2])
        
        x = self.dropout(h)
        x = relu(self.linear(x))
        
        g_in = []
        for batch in x.view(batch_size, n, self.linear_units):
            for i in range(n-1):
                g_in.append(torch.cat((batch[i], batch[i+1])))
        g_in = torch.stack(g_in)

        f = self.f(x)
        g = self.g(g_in)

        f = f.view(batch_size, n, self.classes)
        g = g.view(batch_size, n-1, self.classes, self.classes)
        
        nu_alp = crf.forward_pass(f, g)
        nu_bet = crf.backward_pass(f, g)
        
        x = crf.log_marginal(nu_alp, nu_bet)
        x = torch.exp(x)
        
        return {"p": x, "f": f, "g": g,
                "alpha": nu_alp, "beta": nu_bet}
