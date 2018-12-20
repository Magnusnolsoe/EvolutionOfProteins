from tqdm import tqdm
from data import DataLoader, DataIterator
from utils import build_mask, cosine_similarity


def verbose_train(data_path, model, optimizer, criterion, device, args):
	data_loader = DataLoader(data_path, args.verbose)

	X, y, seq = data_loader.run_pipeline(args.split_rate)
	
	train_iter = DataIterator(X[0], y[0], seq[0], batch_size=args.batch_size)
	test_iter = DataIterator(X[1], y[1], seq[1], batch_size=args.batch_size)
	
	train_err, test_err = [], []
	train_acc, test_acc = [], []
	
	print(model)
	
	for epoch in range(args.epoch):
		
		print("Epoch: {} / {}".format(epoch+1, args.epoch))

		### TRAIN LOOP ###
		err = []
		acc = []
		model.train()
		for proteins, sequence_lengths, targets in tqdm(train_iter, ascii=False, desc="Training", total=int(len(X[0]) / args.batch_size), unit="batch"):

			inputs = proteins.to(device)
			seq_lens = sequence_lengths.to(device)
			targets = targets.to(device)
			
			predictions = model(inputs, seq_lens)
			
			mask = build_mask(sequence_lengths).to(device)

			optimizer.zero_grad()
			batch_loss = criterion(predictions, targets, mask)
			batch_loss.backward()
			optimizer.step()

			cos_sim = cosine_similarity(predictions, targets, mask)
			
			err.append(batch_loss.cpu().item())
			acc.append(cos_sim.cpu().item())

		epoch_trainig_error = sum(err) / len(err)
		epoch_training_accuracy = sum(acc) / len(acc)
		train_err.append(epoch_trainig_error)
		train_acc.append(epoch_training_accuracy)

	
		### TEST LOOP ###
		err = []
		acc = []
		model.eval()
		for proteins, sequence_lengths, targets in tqdm(test_iter, ascii=False, desc="Testing", total=int(len(X[1]) / args.batch_size), unit="batch"):

			inputs = proteins.to(device)
			seq_lens = sequence_lengths.to(device)
			targets = targets.to(device)

			predictions = model(inputs, seq_lens)      
			
			mask = build_mask(sequence_lengths).to(device)

			batch_loss = criterion(predictions, targets, mask)
			
			cos_sim = cosine_similarity(predictions, targets, mask)
			
			err.append(batch_loss.cpu().item())
			acc.append(cos_sim.cpu().item())

		epoch_test_error = sum(err) / len(err)
		epoch_test_accuracy = sum(acc) / len(acc)
		test_err.append(epoch_test_error)
		
		print("Training error: {0:.4f},\tTest error: {1:.4f}\t\tTraining accuracy: {2:.4f}\tTest accuracy: {3:.4f}".format(epoch_trainig_error, epoch_test_error, epoch_training_accuracy, epoch_test_accuracy))
			
	return (train_err, test_err)


def train(data_path, net, optimizer, criterion, device, args):
	
	data_loader = DataLoader(data_path, args.verbose)

	X, y, seq = data_loader.run_pipeline(args.split_rate)
	
	train_iter = DataIterator(X[0], y[0], seq[0], batch_size=args.batch_size)
	test_iter = DataIterator(X[1], y[1], seq[1], batch_size=args.batch_size)
	
	train_err, test_err = [], []
	
	for epoch in range(args.epoch):

		### TRAIN LOOP ###
		err = []
		net.train()
		for proteins, sequence_lengths, targets in train_iter:

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
		for proteins, sequence_lengths, targets in test_iter:

			inputs = proteins.to(device)
			seq_lens = sequence_lengths.to(device)
			targets = targets.to(device)

			predictions = net(inputs, seq_lens)      
			
			mask = build_mask(sequence_lengths).to(device)

			batch_loss = criterion(predictions, targets, mask)

			err.append(batch_loss.cpu().item())

		epoch_test_error = sum(err) / len(err)
		test_err.append(epoch_test_error)
			
	return (train_err, test_err)
# plt.plot(train_err, 'ro-', label="train error")
# plt.plot(test_err, 'bo-', label="test error")
# plt.show()