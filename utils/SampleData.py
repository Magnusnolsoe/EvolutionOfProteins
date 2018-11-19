#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import random
import progressbar
import argparse


args = None


def parseArguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("-n", "--num_samples", help="Enter number of random samples from source file", type=int, default=40000)
	parser.add_argument("-o", "--output", help="Destination path for output file", type=str, default="samples.txt")
	parser.add_argument("-i", "--input", help="Source file with original data.", type=str, default="train.txt")

	args = parser.parse_args()

def extractSamples():
	samples_index = random.sample(range(0, (1726515-1)), args.num_samples)
	samples_index = sorted(samples_index)

	counter = 0
	selected = 0

	widgets = ['Progress: ', progressbar.Percentage(),
				progressbar.Bar(),
				progressbar.ETA()]
	bar = progressbar.ProgressBar(widgets=widgets, max_value=args.num_samples).start()

	with open(args.input, 'r') as f:
	    with open(args.output, 'a') as s:
		    for line in f:
		        if counter in samples_index:
		            s.write(line)
		            selected += 1
		            bar.update(selected)

		        counter += 1


if __name__ == "__main__":
	parseArguments()
	extractSamples()