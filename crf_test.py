#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 12:46:44 2018

@author: magnusnolsoe
"""
import torch
import crf

b = 3
n = 50 # length of sequence
c = 2 # number of classes

g = torch.zeros((b, n-1, c, c)) # independent variables
f = torch.ones((b, n, c)) # this should lead to equal probability for both classes 

y = torch.zeros([b, n], dtype=torch.long) # all label belong to class 1
mask = torch.ones(y.shape, dtype=torch.float)

nu_alp = crf.forward_pass(f, g)
print(nu_alp.shape)
print(nu_alp[0].t())

print()

nu_bet = crf.backward_pass(f, g)
print(nu_bet.shape)
print(nu_bet[0].t())

print("log_likelihood")
print(crf.log_likelihood(y, f, g, nu_alp, nu_bet, mask))
print()

print(crf.logZ( nu_alp, nu_bet)) # should give same log Z no matter what slice we use
print(crf.logZ( nu_alp, nu_bet, 2))
print(crf.logZ( nu_alp, nu_bet, n-1))

print(torch.exp(crf.log_marginal( nu_alp, nu_bet))[0].t())
print(torch.exp(crf.log_marginal( nu_alp, nu_bet, 0))[0])
print(torch.exp(crf.log_marginal( nu_alp, nu_bet, 2))[0])