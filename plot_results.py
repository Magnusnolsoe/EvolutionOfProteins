#!/usr/bin/python3

import matplotlib.pyplot as plt
import pickle
import numpy as np

from os import walk

plt.style.use('ggplot')

filenames = None

result_dir = 'results'

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

training_dict = {}
test_dict = {}

for file in filenames:
	placeholder = pickle.load(open(result_dir + '/' + file, 'rb'))
	training_error = np.array(placeholder[0])
	test_error = np.array(placeholder[1])

	x1 = np.linspace(0.5, 19.5, 20)
	x2 = np.linspace(0.5, 19.5, 20)

	training_epoch = epoch_error(training_error)
	test_epoch = epoch_error(test_error)

	training_dict[file] = training_epoch[-1:][0]
	test_dict[file] = test_epoch[-1:][0]

	plt.plot(x1, training_epoch,  label="training error", c='r', marker='o')
	plt.plot(x2, test_epoch, label="test error", c='b', marker='o')

	x1 = np.linspace(0, 20, len(training_error))
	x2 = np.linspace(0, 20, len(test_error))

	plt.plot(x1, training_error, c='r', linestyle='--', alpha=0.5)
	plt.plot(x2, test_error, c='b', linestyle='--', alpha=0.5)
	
	plt.ylabel("Error")
	plt.xlabel("Epochs")
	plt.xticks(np.linspace(0, 20, 21, dtype=int))
	
	plt.legend(facecolor="#ffffff")
	
	plt.title(', '.join(file.replace('.pk', '').split('-')))
	
	plt.tight_layout()

	plt.savefig(file + '.png', bbox_inches="tight")
	plt.close()


training_rank = [(k, training_dict[k]) for k in sorted(training_dict, key=training_dict.get, reverse=False)]
test_rank = [(k, test_dict[k]) for k in sorted(test_dict, key=test_dict.get, reverse=False)]

for i, (k, v) in enumerate(training_rank):
	if i < 5:
		plt.scatter(i, v, label=', '.join(k.replace('.pk', '').split('-')))
	else:
		break

plt.title("Top 5 Training Error")
plt.ylabel("Error")
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,       # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)

plt.legend()
plt.savefig("Top5Train.png")
plt.close()

for i, (k, v) in enumerate(test_rank):
	if i < 5:
		plt.scatter(i, v, label=', '.join(k.replace('.pk', '').split('-')))
	else:
		break

plt.title("Top 5 Test Error")
plt.ylabel("Error")
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,       # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)

plt.legend()
plt.savefig("Top5Test.png")
plt.close()

lstm_layers = [1, 2, 3]

for layer in lstm_layers:
	for file in filenames:
		if 'lstm_layer=' + str(layer) in file:
			placeholder = pickle.load(open(result_dir + '/' + file, 'rb'))
			training_error = placeholder[0]
			test_error = placeholder[1]

			plt.subplot(2, 1, 1)
			plt.plot(epoch_error(training_error), label=', '.join(file.replace('.pk', '').split('-')))

			plt.subplot(2, 1, 2)
			plt.plot(epoch_error(test_error), label=', '.join(file.replace('.pk', '').split('-')))

	plt.subplot(2, 1, 1)
	plt.title("Training Error")
	plt.ylabel("Error")

	plt.tick_params(
	    axis='x',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    bottom=True,       # ticks along the bottom edge are off
	    top=False,         # ticks along the top edge are off
	    labelbottom=False)

	plt.subplot(2, 1, 2)
	plt.title("Test Error")
	plt.ylabel("Error")
	plt.xlabel("Epochs")

	plt.subplot(2, 1, 1)

	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.savefig('Lstm_layer=' + str(layer) + '.png', bbox_inches="tight")
	plt.close()