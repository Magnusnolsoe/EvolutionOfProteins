# For python2/3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def onehot_tensor(t, num_classes):
    out = np.zeros((t.shape[0], t.shape[1], num_classes))
    for batch in range(t.shape[0]):
        for row, col in enumerate(t[batch]):
            out[batch, row, col] = 1
    return out

def onehot_vector(t, num_classes):
    out = np.zeros((t.shape[0], num_classes))
    for batch in range(t.shape[0]):
        out[batch, t[batch]] = 1
    return out

def log_sum_exp(z, axis=1):
    zmax = np.max(z, axis)
    return zmax + np.log( np.sum( np.exp( z - np.expand_dims(zmax, axis)), axis))
     
def forward_step(nu, f, g, class_dimension):
    return f + log_sum_exp(g + np.expand_dims(nu, class_dimension+1), axis=class_dimension)

def forward_pass(f, g, time_major=False):
    print(f.shape)
    print(g.shape)
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

def log_likelihood(y, f, g, nu_alp, nu_bet, mean_batch=True, time_major=False):
    if not time_major:
        y = np.transpose(y, [1, 0])
        f = np.transpose(f, [1, 0, 2])
        g = np.transpose(g, [1, 0, 2, 3])
    sequence_length = np.shape(f)[0]
    batch_size = np.shape(f)[1]
    f_term = np.zeros((batch_size))
    g_term = np.zeros((batch_size))
    z_term = np.zeros((batch_size))
    for index in range(sequence_length):
        f_term += f[index, np.arange(batch_size), y[index]] # f(y_i,h_i) term
    for index in range(sequence_length - 1):
        g_term += g[index, np.arange(batch_size), y[index + 1], y[index]] # g(y_i,y_i+1,h_i)
    z_term = logZ( nu_alp, nu_bet)
    log_like = f_term + g_term - z_term
    if mean_batch:
        log_like = np.mean(log_like)
    return log_like

def extract_values_vector_f(f, y):
    dim0, dim1, dim2 = f.shape
    idx0 = np.repeat(np.arange(dim0), dim1)
    idx1 = np.asarray([np.arange(dim1)] * dim0).reshape([-1])
    idx2 = y.reshape([-1])
    print(idx0)
    print(idx1)
    print(idx2)
    return f[idx0, idx1, idx2].reshape([dim0, dim1])

def extract_values_vector_g(g, y):
    g_seq_dim, batch_dim, c_dim1, c_dim2 = g.shape
    g_seq_idx = np.repeat(np.arange(g_seq_dim), batch_dim)
    g_batch_idx = np.asarray([np.arange(batch_dim)] * (g_seq_dim)).reshape([-1])
    y1 = y[g_seq_idx, g_batch_idx]
    y2 = y[g_seq_idx+1, g_batch_idx]
    john = g[g_seq_idx, g_batch_idx, y1, y2].reshape([g_seq_dim, batch_dim])
    return g[g_seq_idx, g_batch_idx, y1, y2].reshape([g_seq_dim, batch_dim])

def log_likelihood_vectorized(y, f, g, nu_alp, nu_bet, mean_batch=True, time_major=False):
    if not time_major:
        y = np.transpose(y, [1, 0])
        f = np.transpose(f, [1, 0, 2])
        g = np.transpose(g, [1, 0, 2, 3])
    f_term = np.sum(extract_values_vector_f(f, y), axis=0)
    g_term = np.sum(extract_values_vector_g(g, y), axis=0)
    z_term = logZ(nu_alp, nu_bet)
    log_like = f_term + g_term - z_term
    if mean_batch:
        log_like = np.mean(log_like)
    return log_like


def log_likelihood_hot(y, f, g, nu_alp, nu_bet, mean_batch=True, time_major=False):
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

b = 3
n = 50 # length of sequence
c = 2 # number of classes

g = np.zeros((b, n-1, c, c)) # independent variables
f = np.ones((b, n, c)) # this should lead to equal probability for both classes 

# define some example data that we will use below to calculate the log likelihood
y = np.zeros([b, n], dtype=np.int32) # all label belong to class 1
y_hot = onehot_tensor(y, c)

nu_alp = forward_pass(f, g)
nu_bet = backward_pass( f, g)

print("log_likelihood")
print(log_likelihood(y, f, g, nu_alp, nu_bet))
print()
print("log_likelihood_vectorized")
print(log_likelihood_vectorized(y, f, g, nu_alp, nu_bet))
print()
print(n*np.log(0.5)) # compare to log likelihood for n independent variables with 0.5 probability

print(logZ( nu_alp, nu_bet)) # should give same log Z no matter what slice we use
print(logZ( nu_alp, nu_bet, 2))
print(logZ( nu_alp, nu_bet, n-1))