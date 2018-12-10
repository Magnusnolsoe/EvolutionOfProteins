# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 12:05:54 2018

@author: s144471
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def log_sum_exp(z, axis=1):
    zmax = np.max(z, axis)
    return zmax + np.log( np.sum( np.exp( z - np.expand_dims(zmax, axis)), axis))
     
def forward_step(nu, f, g, class_dimension):
    return f + log_sum_exp(g + np.expand_dims(nu, class_dimension+1), axis=class_dimension)

def forward_pass(f, g, time_major=False):
    if not time_major:
        f = np.transpose(f, [1, 0, 2])
        g = np.transpose(g, [1, 0, 2, 3])
    nu = np.zeros(np.shape(f))
    nu[0] = f[0]
    sequence_length = f.shape[0]
    class_dimension = 1
    for index in range(1, sequence_length):
        fs = forward_step(nu[index-1], f[index], g[index-1], class_dimension)
        nu[index] = fs
    if not time_major:
        nu = np.transpose(nu, [1, 0, 2])
    return nu

def backward_step(nu, f, g, class_dimension):
    return log_sum_exp(np.expand_dims(f + nu, class_dimension) + g, axis=class_dimension+1)

def backward_pass(f, g, time_major=False):
    if not time_major:
        f = np.transpose(f, [1, 0, 2])
        g = np.transpose(g, [1, 0, 2, 3])
    nu = np.zeros(np.shape(f))
    sequence_length = f.shape[0]
    class_dimension = 1
    for index in range(1, sequence_length):
        nu[-index-1] = backward_step(nu[-index], f[-index],g[-index],
                                     class_dimension=class_dimension)
    if not time_major:
        nu = np.transpose(nu, [1, 0, 2])
    return nu

def logZ(nu_alp, nu_bet, index=0, time_major=False):
    if not time_major:
        nu_alp = np.transpose(nu_alp, [1, 0, 2])
        nu_bet = np.transpose(nu_bet, [1, 0, 2])
    class_dimension = 1
    return log_sum_exp(nu_alp[index]+nu_bet[index],
                       axis=class_dimension)

def log_marginal(nu_alp, nu_bet, index=None, time_major=False):
    if not time_major:
        nu_alp = np.transpose(nu_alp, [1, 0, 2])
        nu_bet = np.transpose(nu_bet, [1, 0, 2])
    sequence_length = nu_alp.shape[0]
    if index is None:
        index=np.arange(sequence_length)
    r1 = nu_alp[index]
    r2 = nu_bet[index]
    r3 = np.expand_dims(logZ(nu_alp, nu_bet, time_major=True), axis=1)
    res = r1 + r2 - r3
    if not time_major:
        if len(res.shape) == 3:
            res = np.transpose(res, [1, 0, 2])
    return res