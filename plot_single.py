import matplotlib.pyplot as plt
import numpy as np
import pickle

from os import walk

plt.style.use('ggplot')

result_dir = 'results3'

filenames = None
epochs = None

def epoch_error(seq, num=20):
	avg = len(seq) / float(num)
	out = []
	last = 0.0
	epoch_errors = []

	while last < len(seq):
		out.append(seq[int(last):int(last + avg)])
		last += avg

	for batch_errors in out:
		epoch_errors.append(np.average(batch_errors))

	return epoch_errors


for (dirpath, dirname, filename) in walk(result_dir):
	filenames = filename
	break

with open(result_dir + "/epoch.txt", 'r') as f:
	epochs = int(f.readlines()[0])


for file in filenames:
	if '.pk' in file:
		placeholder = pickle.load(open(result_dir + '/' + file, 'rb'))
		training_error = np.array(placeholder[0])
		test_error = np.array(placeholder[1])

		training_epoch = epoch_error(training_error, epochs)
		test_epoch = epoch_error(test_error, epochs)

		x1 = np.linspace(1, epochs, len(training_epoch))
		x2 = np.linspace(1, epochs, len(test_epoch))

		plt.plot(x1, training_epoch, label="Training Error", marker='o', c='r')
		plt.plot(x2, test_epoch, label="Test Error", marker='o', c='b')

		plt.legend()
		plt.ylabel("Error")
		plt.xlabel("Epoch")

		plt.show()