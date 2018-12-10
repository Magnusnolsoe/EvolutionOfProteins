# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 11:05:10 2018

@author: s144471
"""
import torch
import crf
import numpy as np

from data import DataLoader
from model import Net

dl = DataLoader("data/test.txt")
dl.load_data()

model = Net(torch.device("cpu"), num_embeddings=20, rnn_hidden_size=10)

b = 1
n = len(dl.inputs[0])
c = 20

x = dl.inputs[0].unsqueeze(0)
y = dl.targets[0]
seq_len = [dl.sequence_lengths[0]]

model.eval()
out = model(x, seq_len)

f = out['f']
g = out['g']

nu_alp = crf.forward_pass(f.unsqueeze(0).detach().numpy(), g.detach().numpy().reshape((1,215,20,20)))
nu_bet = crf.backward_pass(f.unsqueeze(0).detach().numpy(), g.detach().numpy().reshape((1,215,20,20)))

print(nu_alp.shape)
print(nu_bet.shape)


#print(np.exp(crf.log_marginal( nu_alp, nu_bet))[0])
