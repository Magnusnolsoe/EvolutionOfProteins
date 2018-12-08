import torch

class Logger:
	def __init__(self, verbose):
		self.verbose = verbose
		
	def info(self, msg):
		if self.verbose:
			print(msg)

def get_num_lines(fname):
	""" Returns the number of lines in a given file. """
	with open(fname) as f:
		for i, l in enumerate(f):
			pass
	return i + 1

def random_guess(seq_lengths):
	
	max_len = seq_lengths[0]
	guess = []
	for seq in seq_lengths:
		random_profile = []
		for i in range(seq):
			uniform = torch.ones(20) / 20
			random_profile.append(uniform)
			
		for i in range(max_len - seq):
			uniform = torch.zeros(20)
			random_profile.append(uniform)
			
		guess.append(torch.stack(random_profile))
		
	return torch.stack(guess)


def build_mask(seq_lengths):
    """ Creates a mask matrix for a batch of sequences. """
    max_len = seq_lengths[0]
    mask = []
    for seq_len in seq_lengths:
        ones_mask = torch.ones(seq_len)
        zeros_mask = torch.zeros(max_len-seq_len)
        mask.append(torch.cat((ones_mask, zeros_mask)))
        
    return torch.stack(mask)

def custom_cross_entropy(batch_pred, batch_target, mask):
    """ Calculates the average cross entropy error over a batch. """
    epsilon=1E-8
    
    CEs = (-(batch_target * torch.log(batch_pred+epsilon))).sum(2)
    
    seq_avg_ce = (CEs*mask).sum(1) / mask.sum(1)

    batch_error = seq_avg_ce.mean(0)
    
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
