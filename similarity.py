from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from data import DataLoader
from utils import build_mask

def compare(data_path):
    data_loader = DataLoader(data_path)
    data_loader.load_data()
    _,t,seq_len = data_loader.sort_data(data_loader.inputs, data_loader.targets, data_loader.sequence_lengths)
    
    
    s = np.zeros((len(t), seq_len[0]))
    
    test_array = np.full((20,), 1/20)
    test_sparse = sparse.csr_matrix(test_array)

    for i,target in enumerate(t) :
        if (i%100==0):
            print(i)
        for j, amino in enumerate(target):
            amino_sparse = sparse.csr_matrix(amino.numpy())
            s[i][j] = cosine_similarity(amino_sparse, test_sparse)
        
   
    mask = build_mask(seq_len)
    
    a = np.zeros(seq_len[0],)
    for i in range(0,seq_len[0]):
        a[i] = sum(s[:,i]) / sum(mask[:,i])

    plt.plot(a)
