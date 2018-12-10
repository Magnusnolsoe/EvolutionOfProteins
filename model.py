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
        
        self.LSTM = nn.LSTM(embedding_dim, hidden_size=rnn_hidden_size, num_layers=rnn_layers,
                           batch_first=True, dropout=rnn_dropout, bidirectional=bi_dir)
        
        self.dropout = nn.Dropout(linear_dropout)
        
        self.f = nn.Linear(self.directions*rnn_hidden_size, linear_out, bias=True)
        
        self.g = nn.Linear(2*self.directions*rnn_hidden_size, linear_out*linear_out, bias=True)
        
    def init_hidden(self, batch_size):
        h0 = torch.randn(self.num_layers*self.directions, batch_size, self.rnn_hidden_size).to(self.device)
        c0 = torch.randn(self.num_layers*self.directions, batch_size, self.rnn_hidden_size).to(self.device)
        
        h0 = Variable(h0)
        c0 = Variable(c0)

        return (h0, c0)

        
    def forward(self, batch, seq_lengths):
        
        batch_size = batch.shape[0]

        embedded_batch = self.embedding(batch)
        
        # Pack sequence
        packed = pack_padded_sequence(embedded_batch, seq_lengths, batch_first=True)
        
        hidden_state_params = self.init_hidden(batch_size)
        rnn_out, (h_n, c_n) = self.LSTM(packed, hidden_state_params)
        
        # Unpack sequence
        h, _ = pad_packed_sequence(rnn_out, batch_first=True, padding_value=0)
        
        # max_seq_len = x.size()[1]
        
        f_in = h.contiguous()
        f_in = f_in.view(-1, h.shape[2])
        
        f_out = self.f(f_in)
        
        g_in = []
        for target in h:
            for i in range(target.shape[0] - 1):
                g_in.append(torch.cat((target[i], target[i+1])))
        g_in = torch.stack(g_in)
        
        g_out = self.g(g_in)
        
        
        return {"f": f_out, "g": g_out}
    
