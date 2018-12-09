import torch
from  torch.optim.lr_scheduler import StepLR

import pickle

from tqdm import tqdm
from data import DataIterator
from utils import pad_predictions



def train(data, net, optimizer, criterion, device, epochs, batch_size, output_dir):
	
	X, y, seq = data

	train_iter = DataIterator(X[0], y[0], seq[0], batch_size=batch_size)
	test_iter = DataIterator(X[1], y[1], seq[1], batch_size=batch_size)

	train_err, test_err = [], []
	    
	scheduler = StepLR(optimizer, step_size=6, gamma=0.9)

	for epoch in range(epochs):

		print("Epoch: " + str(epoch+1) + " / " + str(epochs))
		scheduler.step()

		### TRAIN LOOP ###
		net.train()
		counter = 0
		running_loss = []
		for proteins, sequence_lengths, targets in train_iter:
            
			inputs = proteins.to(device)
			seq_lens = sequence_lengths.to(device)

			predictions = net(inputs, seq_lens)                
			
			targets = targets.to(device)

			batch_loss = criterion(predictions, targets, seq_lens, device)
			running_loss.append(batch_loss.cpu().item())

			optimizer.zero_grad()
			batch_loss.backward()
			optimizer.step()

			counter +=1

			if counter % 200 == 199:
				error = sum(running_loss) / len(running_loss)
				running_loss = []
				train_err.append(error)
				print('Training error: ' + str(error))
				counter = 0

		if len(running_loss) > 0:
			error = sum(running_loss) / len(running_loss)
			train_err.append(error)
			print('Training error: ' + str(error))

                
        ### TEST LOOP ###
		net.eval()
		counter = 0
		running_loss = []
		for proteins, sequence_lengths, targets in test_iter:
            
			inputs = proteins.to(device)
			seq_lens = sequence_lengths.to(device)

			predictions = net(inputs, seq_lens)      
			targets = targets.to(device)

			batch_loss = criterion(predictions, targets, seq_lens, device)
			running_loss.append(batch_loss.cpu().item())
			counter += 1

			if counter % 200 == 199:
				error = sum(running_loss) / len(running_loss)
				running_loss = []
				test_err.append(error)
				print('Test error: ' + str(error))
				counter = 0

		if len(running_loss) > 0:
			error = sum(running_loss) / len(running_loss)
			test_err.append(error)
			print('Test error: ' + str(error))


	final_results = [train_err, test_err]
	file_name = output_dir + "/Emb=" + str(net.embedding_dim) + "-lstm_layer=" + str(net.num_layers) + "-lstm_size=" + str(net.rnn_hidden_size) + ".pk"
	pickle.dump(final_results, open(file_name, "wb"))

                