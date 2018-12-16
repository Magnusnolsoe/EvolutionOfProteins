#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 11:18:08 2018

@author: magnusnolsoe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 11:09:53 2018

@author: magnusnolsoe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 10:15:14 2018

@author: magnusnolsoe
"""

import torch

from model import Net
from data import DataLoader, DataIterator
from utils import custom_cross_entropy as loss, random_guess, build_mask


loader = DataLoader("data/valid.txt", verbose=True)
loader.load_data()

X = loader.inputs
Y = loader.targets
seq_lens = loader.sequence_lengths

X, Y, seq_lens = loader.sort_data(X, Y, seq_lens)

valid_iter = DataIterator(X, Y, seq_lens, batch_size=1)

checkpoint = torch.load("checkpoint/checkpoint.pt", map_location='cpu')

device = torch.device("cpu")

model = Net(device, num_embeddings=21, embedding_dim=100,
                 rnn_hidden_size=512, rnn_layers=2, rnn_dropout=0.35, bi_dir=True,
                 linear_out=20, linear_dropout=0.35).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

pred_err = []
uniform_err = []
for i, (protein, t, profile) in enumerate(valid_iter):
    
    mask = build_mask(t)
    baseline = random_guess(t)
    pred = model(protein, t)
    
    pred_err.append(loss(pred, profile, mask).item())
    uniform_err.append(loss(baseline, profile, mask).item())
    
    if i % 50 == 0:
        print(i)