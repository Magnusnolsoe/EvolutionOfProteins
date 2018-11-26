import torch

def target_to_tensor(batch_targets):
    
    tensor = []
    for target in batch_targets:
        for profile in target:
            tensor.append(profile)
    
    return torch.stack(tensor, dim=0)