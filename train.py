import torch
#import matplotlib.pyplot as plt

from tqdm import tqdm
from data import DataIterator
from utils import pad_predictions
import pickle

def train(data, net, optimizer, criterion, device, epochs, batch_size):
	
	X, y, seq = data

	train_iter = DataIterator(X[0], y[0], seq[0], batch_size=batch_size)
	test_iter = DataIterator(X[1], y[1], seq[1], batch_size=batch_size)

	train_err, test_err = [], []
    
	for epoch in range(epochs):

		print("Epoch: " + str(epoch+1) + " / " + str(epochs))

		### TRAIN LOOP ###
		err = []
		net.train()
		for proteins, sequence_lengths, targets in tqdm(train_iter, ascii=False, desc="Training", total=int(len(X[0]) / batch_size), unit="batch"):
            
			inputs = proteins.to(device)
			seq_lens = sequence_lengths.to(device)

			predictions = net(inputs, seq_lens)                
			
			targets = targets.to(device)

			batch_loss = criterion(predictions, targets, seq_lens, device)
			optimizer.zero_grad()
			batch_loss.backward()
			optimizer.step()

			err.append(batch_loss.cpu().item())
        
		epoch_trainig_error = sum(err) / len(err)
		train_err.append(epoch_trainig_error)

                
        ### TEST LOOP ###
		err = []
		net.eval()
		for proteins, sequence_lengths, targets in tqdm(test_iter, ascii=False, desc="Testing", total=int(len(X[1]) / batch_size), unit="batch"):
            
			inputs = proteins.to(device)
			seq_lens = sequence_lengths.to(device)

			predictions = net(inputs, seq_lens)      
			targets = targets.to(device)

			batch_loss = criterion(predictions, targets, seq_lens, device)

			err.append(batch_loss.cpu().item())

		epoch_test_error = sum(err) / len(err)
		test_err.append(epoch_test_error)

		print("Training error: " + str(epoch_trainig_error))
		print("Test error: " + str(epoch_test_error))

	final_results = [train_err, test_err]
	file_name = "results/Emb=" + str(net.embedding_dim) + "-lstm_layer=" + str(net.num_layers) + "-lstm_size=" + str(net.rnn_hidden_size) + ".pk"
	pickle.dump(final_results, open(file_name, "wb"))

#	plt.plot(train_err, 'ro-', label="train error")
#	plt.plot(test_err, 'bo-', label="test error")
#	plt.legend()
#	plt.title("Emb=" + str(net.embedding_dim) + ", lstm_layer=" + str(net.num_layers) + ", lstm_size=" + str(net.rnn_hidden_size))
#	plt.savefig("Emb=" + str(net.embedding_dim) + ", lstm_layer=" + str(net.num_layers) + ", lstm_size=" + str(net.rnn_hidden_size) + ".png")
	#plt.show()
                