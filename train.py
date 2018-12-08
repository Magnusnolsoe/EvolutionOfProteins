from tqdm import tqdm
from data import DataLoader, DataIterator
from utils import build_mask

def train(data_path, net, optimizer, criterion, device, logger, args):
	
	data_loader = DataLoader(data_path, args.verbose)

	X, y, seq = data_loader.run_pipeline(args.split_rate)
	
	train_iter = DataIterator(X[0], y[0], seq[0], batch_size=args.batch_size)
	test_iter = DataIterator(X[1], y[1], seq[1], batch_size=args.batch_size)
	
	train_err, test_err = [], []
	
	logger.info(net)
	
	for epoch in range(args.epoch):
		
		logger.info("Epoch: {} / {}".format(epoch+1, args.epoch))

		### TRAIN LOOP ###
		err = []
		net.train()
		for proteins, sequence_lengths, targets in (tqdm(train_iter, ascii=False, desc="Training", total=int(len(X[0]) / args.batch_size), unit="batch") if args.verbose else train_iter):

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
		for proteins, sequence_lengths, targets in (tqdm(test_iter, ascii=False, desc="Testing", total=int(len(X[1]) / args.batch_size), unit="batch") if args.verbose else test_iter):

			inputs = proteins.to(device)
			seq_lens = sequence_lengths.to(device)
			targets = targets.to(device)

			predictions = net(inputs, seq_lens)      
			
			mask = build_mask(sequence_lengths).to(device)

			batch_loss = criterion(predictions, targets, mask)

			err.append(batch_loss.cpu().item())

		epoch_test_error = sum(err) / len(err)
		test_err.append(epoch_test_error)
		
		logger.info("Training error: {0:.4f},\tTest error: {0:.4f}".format(epoch_trainig_error, epoch_test_error))
			
	return (train_err, test_err)
# plt.plot(train_err, 'ro-', label="train error")
# plt.plot(test_err, 'bo-', label="test error")
# plt.show()