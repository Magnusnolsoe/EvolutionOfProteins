#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 19:04:12 2018

@author: magnusnolsoe
"""

import torch
import numpy as np

def log_sum_exp(z, axis=1):
    zmax = torch.max(z, axis)[0]
    return zmax + torch.log( torch.sum( torch.exp( z - torch.unsqueeze(zmax, axis)), axis))

def forward_step(nu, f, g, class_dimension):
    return f + log_sum_exp(g + torch.unsqueeze(nu, class_dimension+1), axis=class_dimension)

def forward_pass(f, g, time_major=False):
    if not time_major:
        f = f.permute(1, 0, 2)
        g = g.permute(1, 0, 2, 3)
    nu = torch.zeros(f.size())
    nu[0] = f[0]
    sequence_length = f.size()[0]
    class_dimension = 1
    for index in range(1, sequence_length):
        fs = forward_step(nu[index-1], f[index], g[index-1], class_dimension)
        nu[index] = fs
    if not time_major:
        nu = nu.permute(1, 0, 2)
    return nu

def backward_step(nu, f, g, class_dimension):
    return log_sum_exp(torch.unsqueeze(f + nu, class_dimension) + g, axis=class_dimension+1)

def backward_pass(f, g, time_major=False):
    if not time_major:
        f = f.permute(1, 0, 2)
        g = g.permute(1, 0, 2, 3)
    nu = torch.zeros(f.size())
    sequence_length = f.size()[0]
    class_dimension = 1
    for index in range(1, sequence_length):
        nu[-index-1] = backward_step(nu[-index], f[-index],g[-index],
                                     class_dimension=class_dimension)
    if not time_major:
        nu = nu.permute(1, 0, 2)
    return nu

def logZ(nu_alp, nu_bet, index=0, time_major=False):
    if not time_major:
        nu_alp = nu_alp.permute(1, 0, 2)
        nu_bet = nu_bet.permute(1, 0, 2)
    class_dimension = 1
    return log_sum_exp(nu_alp[index]+nu_bet[index],
                       axis=class_dimension)
    
def log_likelihood(y, f, g, nu_alp, nu_bet, mask, mean_batch=True, time_major=False):
    if not time_major:
        y = y.permute(1, 0)
        f = f.permute(1, 0, 2)
        g = g.permute(1, 0, 2, 3)
        mask = mask.permute(1, 0)
    sequence_length = f.size()[0]
    batch_size = f.size()[1]
    f_term = torch.zeros((batch_size))
    g_term = torch.zeros((batch_size))
    z_term = torch.zeros((batch_size))
    for index in range(sequence_length):
        idx_sel = mask[index].type(torch.long)
        f_term += f[index, torch.arange(batch_size, dtype=torch.long), idx_sel*y[index]]*mask[index] # f(y_i,h_i) term
    for index in range(sequence_length - 1):
        idx_sel1 = mask[index + 1].type(torch.long)
        idx_sel2 = mask[index].type(torch.long)
        g_term += g[index, torch.arange(batch_size, dtype=torch.long), y[index + 1]*idx_sel1, y[index]*idx_sel2]*(mask[index]*mask[index+1]) # g(y_i,y_i+1,h_i)
    z_term = logZ( nu_alp, nu_bet)
    log_like = f_term + g_term - z_term
    if mean_batch:
        log_like = torch.mean(log_like)
    return log_like

def log_marginal(nu_alp, nu_bet, index=None, time_major=False):
    if not time_major:
        nu_alp = nu_alp.permute(1, 0, 2)
        nu_bet = nu_bet.permute(1, 0, 2)
    sequence_length = nu_alp.size()[0]
    if index is None:
        index=torch.arange(sequence_length, dtype=torch.long)
    r1 = nu_alp[index]
    r2 = nu_bet[index]
    r3 = torch.unsqueeze(logZ(nu_alp, nu_bet, time_major=True), dim=1)
    res = r1 + r2 - r3
    if not time_major:
        if len(res.size()) == 3:
            res = res.permute(1, 0, 2)
    return res