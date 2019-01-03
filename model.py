import torch.nn as nn
import torch
import crf

from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import relu, sigmoid, softmax

class BiRNN_CRF(nn.Module):
    def __init__(self, device, num_embeddings=21, embedding_dim=32,
                 rnn_hidden_size=100, rnn_layers=2, rnn_dropout=0.3, bi_dir=True,
                 hidden_linear_units=200, classes=20, linear_dropout=0.5, crf_on=False):
        super(BiRNN_CRF, self).__init__()
        
        self.device = device

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        self.rnn_hidden_size = rnn_hidden_size
        self.num_layers = rnn_layers
        self.directions= 2 if bi_dir else 1
        self.classes = classes
        self.linear_units = hidden_linear_units
        self.crf_on = crf_on
        rnn_dropout = rnn_dropout if rnn_layers > 1 else 0
        
        self.LSTM = nn.LSTM(embedding_dim, hidden_size=rnn_hidden_size, num_layers=rnn_layers,
                           batch_first=True, dropout=rnn_dropout, bidirectional=bi_dir)
        
        self.dropout = nn.Dropout(linear_dropout)
        
        self.linear_hidden = nn.Linear(self.directions*rnn_hidden_size, hidden_linear_units)
        
        self.f = nn.Linear(hidden_linear_units, classes, bias=True)
        
        if crf_on:
            self.g = nn.Linear(2*hidden_linear_units, classes**2, bias=True)
        

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
        x = relu(self.linear_hidden(x))
        x = self.dropout(x)
        
        f = self.f(x)
        f = f.view(batch_size, n, self.classes)
        
        if self.crf_on:
            
            g_in = []
            for batch in x.view(batch_size, n, self.linear_units):
                for i in range(n-1):
                    g_in.append(torch.cat((batch[i], batch[i+1])))
            g_in = torch.stack(g_in)
    
            g = self.g(g_in)
            g = g.view(batch_size, n-1, self.classes, self.classes)
            
            nu_alp = crf.forward_pass(f, g, self.device)
            nu_bet = crf.backward_pass(f, g, self.device)
            
            log_pred = crf.log_marginal(nu_alp, nu_bet)
            prediction = torch.exp(log_pred)
            
        else:
            prediction = softmax(f, dim=2)
            
        return prediction
