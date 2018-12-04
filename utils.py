import torch
from torch import t as T

def get_num_lines(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def pad_profiles(batch, seq_lengths):
    
    max_len = seq_lengths[0]
    
    padded_batch = []
    for profile, seq_len in zip(batch, seq_lengths):
        padded_batch.append(T(torch.nn.functional.pad(T(profile),
                                                      (0, max_len-seq_len),
                                                      value=1)))

    return torch.stack(padded_batch)
        
def build_mask(seq_lengths):
    
    max_len = seq_lengths[0]
    mask = []
    for seq_len in seq_lengths:
        ones_mask = torch.ones(seq_len)
        zeros_mask = torch.zeros(max_len-seq_len)
        mask.append(torch.cat((ones_mask, zeros_mask)))
        
    return torch.stack(mask)

def custom_cross_entropy(batch_pred, batch_target, mask):
    
    epsilon=1E-8
    
    CEs = (-(batch_target * torch.log(batch_pred+epsilon))).sum(2)
    
    seq_avg = (CEs*mask).sum(1) / mask.sum(1)

    batch_error = seq_avg.mean(0)
    
    return batch_error

'''
def get_variable(x):
    """ Converts tensors to cuda, if available. """
    if use_cuda:
        return x.cuda()
    return x

def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if use_cuda:
        return x.cpu().data.numpy()
    return x.data.numpy
'''
