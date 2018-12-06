from tqdm import tqdm
from data import DataLoader, DataIterator
from utils import build_mask

def train(data_path, net, optimizer, criterion, device, epochs, batch_size, split_rate=0.33, verbose=False):

	data_loader = DataLoader(data_path, verbose)

	X, y, seq = data_loader.run_pipeline(split_rate)
	
	train_iter = DataIterator(X[0], y[0], seq[0], batch_size=batch_size)
	test_iter = DataIterator(X[1], y[1], seq[1], batch_size=batch_size)

	train_err, test_err = [], []

	for epoch in range(epochs):
		if verbose:
			print("Epoch: {} / {}".format(epoch+1, epochs))

		### TRAIN LOOP ###
		err = []
		net.train()
		for proteins, sequence_lengths, targets in (tqdm(train_iter, ascii=False, desc="Training", total=int(len(X[0]) / batch_size), unit="batch") if verbose else train_iter):

			inputs = proteins.to(device)
			seq_lens = sequence_lengths.to(device)
			targets = targets.to(device)
			
			predictions = net(inputs, seq_lens)                
			
			mask = build_mask(sequence_lengths).to(device)

			optimizer.zero_grad()
			batch_loss = criterion(predictions, targets, mask)
			batch_loss.backward()
			optimizer.step()

			err.append(batch_loss.cpu().item())

		epoch_trainig_error = sum(err) / len(err)
		train_err.append(epoch_trainig_error)

	
		### TEST LOOP ###
		err = []
		net.eval()
		for proteins, sequence_lengths, targets in (tqdm(test_iter, ascii=False, desc="Testing", total=int(len(X[1]) / batch_size), unit="batch") if verbose else test_iter):

			inputs = proteins.to(device)
			seq_lens = sequence_lengths.to(device)
			targets = targets.to(device)

			predictions = net(inputs, seq_lens)      
			
			mask = build_mask(sequence_lengths).to(device)

			batch_loss = criterion(predictions, targets, mask)

			err.append(batch_loss.cpu().item())

		epoch_test_error = sum(err) / len(err)
		test_err.append(epoch_test_error)

		print("Training error: {0:.4f},\tTest error: {0:.4f}".format(epoch_trainig_error, epoch_test_error))

# plt.plot(train_err, 'ro-', label="train error")
# plt.plot(test_err, 'bo-', label="test error")
# plt.show()