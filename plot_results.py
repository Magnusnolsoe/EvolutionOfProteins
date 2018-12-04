import matplotlib.pyplot as plt
import pickle

from os import walk

filenames = None

for (dirpath, dirname, filename) in walk('results'):
	filenames = filename
	break

for file in filenames:
	placeholder = pickle.load(open('results/' + file, 'rb'))
	training_error = placeholder[0]
	test_error = placeholder[1]

	plt.plot(training_error, label="training error")
	plt.plot(test_error, label="test error")
	plt.legend()
	plt.title(', '.join(file.replace('.pk', '').split('-')))
	plt.savefig(file + '.png')
	plt.close()