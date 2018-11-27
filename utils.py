import torch
from torch import t as T
from torch.autograd import Variable

def target_to_tensor(batch_targets):
    
    tensor = []
    for target in batch_targets:
        for profile in target:
            tensor.append(profile)
    
    return torch.stack(tensor, dim=0)

def pad_targets(batch, seq_lengths):
    
    max_len = max(seq_lengths)
    
    padded_batch = []
    mask_batch = []
    for profile, seq_len in zip(batch, seq_lengths):
        padded_batch.append(T(torch.nn.functional.pad(T(profile),
                                                      (0, max_len-seq_len),
                                                      value=1)))
        ones_mask = torch.ones(profile.shape)
        zeros_mask = torch.zeros((max_len-seq_len, 20))
        mask_batch.append(torch.cat((ones_mask, zeros_mask)))
        
    return torch.stack(padded_batch), torch.stack(mask_batch)
        

def custom_cross_entropy(batch_y, batch_t, seq_len):
    
    y_padded, mask = pad_targets(batch_y, seq_len)
    t_padded, _ = pad_targets(batch_t, seq_len)
    
    
    
