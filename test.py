from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import conditional_random_fields as crf

def onehot_tensor(t, num_classes):
    print(t)
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

b = 3
n = 50 # length of sequence
c = 2 # number of classes

g = np.zeros((c,c,n-1)) # independent variables
f = np.ones((c,n)) # this should lead to equal probability for both classes 

# define some example data that we will use below to calculate the log likelihood
y = np.zeros([b, n], dtype=np.int32) # all label belong to class 1
y = onehot_tensor(y, c)

print("test forward_pass ...")
nu_alp = crf.forward_pass(f, g)
print(nu_alp)
print()

print("test backward_pass ...")
nu_bet = crf.backward_pass(f, g)
print(nu_bet)
print()


print("test log_likelihood ...")
log_like_test = crf.log_likelihood(y, f, g, nu_alp, nu_bet)
print(log_like_test)
print()

print("test logZ ...")
print(crf.logZ(nu_alp, nu_bet)) # should give same log Z no matter what slice we use
print(crf.logZ(nu_alp, nu_bet, 2))
print(crf.logZ(nu_alp, nu_bet, n-1))

print("test log_marginal ...")
print(np.exp(crf.log_marginal(nu_alp, nu_bet)))
print(np.exp(crf.log_marginal(nu_alp, nu_bet, 0)))
print(np.exp(crf.log_marginal(nu_alp, nu_bet, 2)))
print()

print("Example 2: Vertibi decoding (max sum algorithm) ...")
n = 3 # length of sequence
c = 2 # number of classes

g = np.zeros((c,c,n-1)) # independent variables
f = np.zeros((c,n)) #  

f[0,:] = np.log(3) 
f[1,:] = np.log(1) # this should give probability 3 / (3 + 1 ) = 3/4 for class 1

# define some example data that we will use below to calculate log likelihood
y = np.zeros([n], dtype=np.int32) # all label belong to class 1
y = onehot_tensor(y, c)

nu_alp = crf.forward_pass(f, g)
nu_bet = crf.backward_pass(f, g)

print(np.exp(nu_alp))
print(np.exp(nu_bet))

print(crf.logZ(nu_alp, nu_bet)) # should give same log Z no matter what slice we use
print(crf.logZ(nu_alp, nu_bet, 2))
print(crf.logZ(nu_alp, nu_bet, n-1))
print(n*np.log(4)) # exact log Z

print(crf.log_likelihood(y, f, g, nu_alp, nu_bet))
print(n*np.log(3.0/4.0)) # compare to independent
print(np.exp(crf.log_marginal( nu_alp, nu_bet)))
print()

print("Example 3 ...")
g[0,0,:] = np.ones([n-1])

print(g[:,:,0])
print(g[:,:,1])

nu_alp = crf.forward_pass( f, g)
nu_bet = crf.backward_pass( f, g)

print(np.exp(nu_alp))
print(np.exp(nu_bet))

# should give same log Z no matter what slice we use
print(crf.logZ(nu_alp, nu_bet)) 
print(crf.logZ(nu_alp, nu_bet, 1))
print(crf.logZ(nu_alp, nu_bet, n-1))
# exact log Z
print(np.log(np.power(3,n)*np.power(np.exp(1),n-1)+2*np.power(3,n-1)*
             np.power(np.exp(1),n-2)+np.power(3,n-1)+3*np.power(3,n-2)+1))

print(crf.log_likelihood(y, f, g, nu_alp, nu_bet))
print(n*np.log(3.0/4.0)) # compare to independent
print(np.exp(crf.log_marginal( nu_alp, nu_bet)))
