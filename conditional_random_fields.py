from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def log_sum_exp( z, axis=0):
    zmax = np.max( z, axis)
    return zmax + np.log( np.sum( np.exp( z - np.expand_dims(zmax, axis)), axis ))
     
def forward_step( nu, f, g):
    return f + log_sum_exp(g + np.expand_dims(nu, 1))

def forward_pass( f, g):
    nu = np.zeros(np.shape(f))
    nu[:,0] = f[:,0]
    for index in range(1,len(f[0,:])):
        nu[:,index] = forward_step( nu[:,index-1], f[:,index], g[:,:,index-1])
    return nu


def backward_step( nu, f, g):
    return log_sum_exp( np.expand_dims(f+nu, 0) + g, axis=1)

def backward_pass( f, g):
    nu = np.zeros(np.shape(f))
    for index in range(1,len(f[0,:])):
        nu[:,-index-1] = backward_step( nu[:,-index], f[:,-index],g[:,:,-index])
    return nu

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


def logZ( nu_alp, nu_bet, index=0):
    return log_sum_exp(nu_alp[:,index]+nu_bet[:,index]) 

def log_likelihood(y, f, g, nu_alp, nu_bet, mean_batch=True, time_major=False):
    if time_major:
        y = np.transpose(y, [1, 0, 2])
        f = np.transpose(f, [1, 0, 2])
        g = np.transpose(g, [1, 0, 2, 3])
    f_term = np.sum(f*y, axis=(1, 2))
    y_i = np.expand_dims(y[:, :-1], axis=3)
    y_plus = np.expand_dims(y[:, 1:], axis=2)
    g_term = np.sum(g*y_i*y_plus, axis=(1,2,3))
    z_term = logZ(nu_alp, nu_bet)
    log_like = f_term + g_term - z_term
    if mean_batch:
        log_like = np.mean(log_like)
    return log_like

def log_marginal( nu_alp, nu_bet, index=None):
    if index is None:
        index=np.asarray(list(range(len(nu_alp[0,:]))))
    return nu_alp[:,index] + nu_bet[:,index] - logZ( nu_alp, nu_bet)
