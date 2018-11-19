class Net(nn.Module):
    def __init__(self, emb_out_size, fc1_out_size, LSTM_out_size):
        super(Net, self).__init__()
        
        self.num_input = 1
        self.num_output = 20
        
        self.embedding_out_size = emb_out_size
        self.LSTM_out = LSTM_out_size
        
        self.hidden_dim = 20
        self.hidden = self.init_hidden()
        
        
        self.embeddings = nn.Embedding(
            self.num_input,
            self.embedding_out_size
        )
        
        self.LSTM = nn.LSTM(
            input_size=self.embedding_out_size,
            hidden_size=self.LSTM_out,
            num_layers=1,
            dropout=0.05,
            bidirectional=True,
            batch_first=True
        )
        
        self.fc = nn.Linear(
            in_features=LSTM_out_size,
            out_features=self.num_output
        )
        
        self.dropout = nn.Dropout(0.3)
        
        
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers * num_directions, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(2, 1, self.hidden_dim)),   
                autograd.Variable(torch.zeros(2, 1, self.hidden_dim)))
        
    def forward(self, x):
        
        x = self.embeddings(x)
        
        x = self.fc1(x)
        x = relu(x)
        x = self.dropout(x)
        
        x, self.hidden = self.lstm(x.view(len(sentence), 1, -1), self.hidden)
        
        x = self.fc2(x)
        x = relu(x)
        x = self.dropout(x)
        
        return x