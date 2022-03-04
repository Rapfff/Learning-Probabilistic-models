import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, stdev
from scipy.signal import hilbert
from math import exp, sqrt
from experiment.nox.edfreader import EDFreader
from src.tools import saveSet, loadSet, setFromList, randomProbabilities
from src.learning.BW_HMM import BW_HMM
from examples.examples_models import modelHMM_random
from random import shuffle, uniform
from pyts.approximation import SymbolicFourierApproximation


EDF_FILE   = "eh_20210211.edf"
EVENT_FILE = "Event_Grid.csv"
SIZE_ALPHABET = 10
#SIGNALS
#HEART_RATE = 52
#ACTIVITY   = 13
#F3_M2 = 44
SIGNAL_ID = 44
SIGNAL_NAME = "F3_M2"

WINDOW_SIZE_SEC_MAX = 10

NB_STATES = 5

def normalize(seq):
	mu  = mean(seq)
	std = stdev(seq)
	return [(i-mu)/std for i in seq]


def read_files():
	events = open(EVENT_FILE,'r')
	l = events.readline()
	l = events.readline()
	l = events.readline()

	r = EDFreader(EDF_FILE)
	length = r.getTotalSamples(SIGNAL_ID)
	frequency = r.getSampleFrequency(SIGNAL_ID)
	exp_duration = length/frequency #in seconds

	start_time_event = [int(i[:2]) for i in l.split(';')[2].split(':')]
	start_time_edf   = [r.getStartTimeHour(),r.getStartTimeMinute(),r.getStartTimeSecond()]
	diff = (start_time_event[0] - start_time_edf[0])*60*60 + (start_time_event[1] - start_time_edf[1])*60 + (start_time_event[2] - start_time_edf[2])
	begining = int(diff*frequency)
	#begining = 4134971
	r.fseek(SIGNAL_ID,begining,EDFreader.EDFSEEK_SET)
	
	data = []
	stages = []
	x_stages = []
	while l:
		l = l.split(';')
		l[1] = int(l[1])
		s = l[0]
		if not s in stages:
			stages.append(s)

		nb_windows = max(l[1]//WINDOW_SIZE_SEC_MAX,1)
		windows_sizes = [ l[1]//nb_windows for i in range(nb_windows) ]
		diff = l[1] - sum(windows_sizes)
		for i in range(diff):
			windows_sizes[i] += 1

		for size in windows_sizes:		
			window = np.arange(int(size*frequency),dtype=np.float_)
			r.readSamples(SIGNAL_ID,window,size)
			data.append([i for i in window])
			x_stages.append(s)

		l = events.readline()


	#while r.ftell(SIGNAL_ID)+window_size < length:
	
	events.close()
	return [data,stages,x_stages]

	
def write_training_test_set(fraction_test,name='',n_coefs=6,n_bins=6):
	"""name is the name of the output files,
	fraction_test is a float between ]0,1[ corresponding to the fraction of sequences in the test set """
	if name != '':
		name = '_'+str(name)
	data, stages, x_stages = read_files()
	transformer = SymbolicFourierApproximation(n_coefs=n_coefs,n_bins=n_bins)
	data = transformer.fit_transform(data)


	"""
	coords = [[] for i in stages]
	set_seq = []
	for i in range(len(data)):
		seq = list(data[i])
		if not seq in set_seq:
			set_seq.append(seq)
			for s in range(len(stages)):
				coords[s].append(0)
		j = set_seq.index(seq)
		coords[stages.index(x_stages[i])][j] += 1
	dist = []
	for s in range(len(stages)-1):
		#dist.append([])
		for ss in range(s+1,len(stages)):
			print(stages[s], " - ",stages[ss], " ",sqrt(sum([(coords[s][i]-coords[ss][i])**2 for i in range(len(set_seq))])))
			#dist[-1].append(sqrt(sum([(coords[s][i]-coords[ss][i])**2 for i in range(len(set_seq))])))
	"""

	l = [i for i in range(len(data))]
	data_shuffled = [ data[i] for i in l]
	x_stages_shuffled = [ x_stages[i] for i in l]


	test_seqs = data_shuffled[:int(fraction_test*len(data_shuffled))]
	training_seqs = data_shuffled[int(fraction_test*len(data_shuffled)):]

	training_set = setFromList(training_seqs)
	test_set = setFromList(test_seqs)

	saveSet(training_set,"training_set"+name+".txt")
	saveSet(test_set,"test_set"+name+".txt")


n_bins = 6
#write_training_test_set(0.1,n_bins=n_bins)

tr = loadSet("training_set.txt")
ts = loadSet("test_set.txt")

#rm = modelHMM_random(5,[chr(i) for i in range(97,97+n_bins)])
#rm.save("init_model.txt")
rm = loadHMM("init_model.txt")

algo = BW_HMM(rm)
out = algo.learn(tr,verbose=True)
out.pprint()

print("Loglikelihood on test_set for initial model ",rm.logLikelihood(ts))
print("Loglikelihood on test_set for output  model ",out.logLikelihood(ts))
