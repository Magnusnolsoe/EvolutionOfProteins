import matplotlib.pyplot as plt
import pickle
import numpy as np

from os import walk

plt.style.use('ggplot')

filenames = None

result_dir = 'results'

for (dirpath, dirname, filename) in walk(result_dir):
	filenames = filename
	break


for file in filenames:
	placeholder = pickle.load(open(result_dir + '/' + file, 'rb'))
	training_error = placeholder[0]
	test_error = placeholder[1]

	x1 = np.linspace(0, 20, len(training_error))
	x2 = np.linspace(0, 20, len(test_error))

	plt.plot(x1, training_error, label="training error")
	plt.plot(x2, test_error, label="test error")
	
	plt.ylabel("Error")
	plt.xlabel("Epochs")
	plt.xticks(np.linspace(0, 20, 21, dtype=int))
	
	plt.legend(facecolor="#ffffff")
	
	plt.title(', '.join(file.replace('.pk', '').split('-')))
	
	plt.tight_layout()

	plt.savefig(file + '.png', bbox_inches="tight")
	plt.close()


lstm_layers = [1, 2, 3]

for layer in lstm_layers:
	for file in filenames:
		if 'lstm_layer=' + str(layer) in file:
			placeholder = pickle.load(open(result_dir + '/' + file, 'rb'))
			training_error = placeholder[0]
			test_error = placeholder[1]

			x1 = np.linspace(0, 20, len(training_error))
			x2 = np.linspace(0, 20, len(test_error))

			plt.subplot(2, 1, 1)
			plt.plot(x1, training_error, label=', '.join(file.replace('.pk', '').split('-')))

			plt.subplot(2, 1, 2)
			plt.plot(x2, test_error, label=', '.join(file.replace('.pk', '').split('-')))

	plt.subplot(2, 1, 1)
	plt.title("Training Error")

	plt.subplot(2, 1, 2)
	plt.title("Test Error")

	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.savefig('Lstm_layer=' + str(layer) + '.png', bbox_inches="tight")
	plt.close()