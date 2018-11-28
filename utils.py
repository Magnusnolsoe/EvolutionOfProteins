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
    for profile, seq_len in zip(batch, seq_lengths):
        padded_batch.append(T(torch.nn.functional.pad(T(profile),
                                                      (0, max_len-seq_len),
                                                      value=1)))
        
    return torch.stack(padded_batch)
        
def build_mask(seq_lengths):
    max_len = max(seq_lengths)
    mask = []
    for seq_len in seq_lengths:
        ones_mask = torch.ones(seq_len)
        zeros_mask = torch.zeros(max_len-seq_len)
        mask.append(torch.cat((ones_mask, zeros_mask)))
        
    return torch.stack(mask)

def custom_cross_entropy(batch_size, batch_y, batch_t, seq_len):
    
    y_padded = pad_targets(batch_y, seq_len)
    t_padded = pad_targets(batch_t, seq_len)
    
    mask = build_mask(seq_len)
    
    threshold=0.0000001
    
    CEs = (-(y_padded * torch.log(t_padded+threshold))).sum(2)
    
    seq_avg = ((CEs)*mask).sum(1) / mask.sum(1)

    batch_error = seq_avg.sum(0) / batch_size
    
    return batch_error
    
    
    
    
