import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from data import DataLoader, DataIterator
from utils import pad_predictions

def train(data_path, net, optimizer, criterion, device, epochs, batch_size, split_rate=0.33):
    
	data_loader = DataLoader(data_path)

	X, y, seq = data_loader.run_pipline(split_rate)

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
			predictions = torch.split(predictions, seq_lens.tolist())

			padded_pred = pad_predictions(predictions, seq_lens)
			padded_pred = padded_pred.to(device)
			targets = targets.to(device)

			batch_loss = criterion(padded_pred, targets, seq_lens, device)
			batch_loss.backward()
			optimizer.step()

			err.append(batch_loss.cpu().item())
        
		epoch_trainig_error = sum(err) / len(err)
		train_err.append(epoch_trainig_error)

                
        ### TEST LOOP ###
		err = []
		net.eval()
		for proteins, sequence_lengths, targets in tqdm(test_iter, ascii=True, desc="Testing", total=int(len(X[1]) / batch_size)):
            
			inputs = proteins.to(device)
			seq_lens = sequence_lengths.to(device)

			predictions = net(inputs, seq_lens)                
			predictions = torch.split(predictions, seq_lens.tolist())

			padded_pred = pad_predictions(predictions, seq_lens)
			padded_pred = padded_pred.to(device)
			targets = targets.to(device)

			batch_loss = criterion(padded_pred, targets, seq_lens, device)

			err.append(batch_loss.cpu().item())

		epoch_test_error = sum(err) / len(err)
		test_err.append(epoch_test_error)

		print("Training error: " + str(epoch_trainig_error))
		print("Test error: " + str(epoch_test_error))

    # plt.plot(train_err, 'ro-', label="train error")
    # plt.plot(test_err, 'bo-', label="test error")
    # plt.show()
                