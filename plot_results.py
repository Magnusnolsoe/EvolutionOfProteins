import matplotlib.pyplot as plt
import pickle
import numpy as np

from os import walk

plt.style.use('ggplot')

filenames = None

for (dirpath, dirname, filename) in walk('results'):
	filenames = filename
	break


for file in filenames:
	placeholder = pickle.load(open('results/' + file, 'rb'))
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