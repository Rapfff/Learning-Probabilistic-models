import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

import numpy as np
import matplotlib.pyplot as plt
from pyts.approximation import SymbolicFourierApproximation
from random import shuffle
from itertools import product
from statistics import mean, stdev

from experiment.nox.edfreader import EDFreader
from src.tools import saveSet, loadSet, setFromList
from src.learning.BW_HMM import BW_HMM
from src.models.HMM import loadHMM
from examples.examples_models import modelHMM_random


EDF_FILE   = "eh_20210211.edf"
EVENT_FILE = "Event_Grid.csv"
SIZE_ALPHABET = 10
#SIGNALS
#HEART_RATE = 52
#ACTIVITY   = 13
#F3_M2 = 44
SIGNAL_ID = 44
SIGNAL_NAME = "F3_M2"

WINDOW_SIZE_SEC_MAX = 10 #nb of sec as input to DFA
NB_WINDOWS_BY_SEQ = 6 #nb of sec by seuquence = WINDOW_SIZE_SEC_MAX*NB_WINDOW_BY_SEQ

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

	
def write_training_test_set(fraction_test,name='',n_coefs=4,n_bins=6):
	"""name is the name of the output files,
	fraction_test is a float between ]0,1[ corresponding to the fraction of sequences in the test set """
	if name != '':
		name = '_'+str(name)
	data, stages, x_stages = read_files()
	transformer = SymbolicFourierApproximation(n_coefs=n_coefs,n_bins=n_bins)
	data = transformer.fit_transform(data)

	data = [''.join(i) for i in data]
	new_data = []
	new_x_stages = []
	for i in range(0,len(data) - NB_WINDOWS_BY_SEQ,NB_WINDOWS_BY_SEQ):
		new_data.append([data[i+j] for j in range(NB_WINDOWS_BY_SEQ)])
		new_x_stages.append([x_stages[i+j] for j in range(NB_WINDOWS_BY_SEQ)])
	new_data.append([data[i+j] for j in range(len(data)%NB_WINDOWS_BY_SEQ)])
	new_x_stages.append([x_stages[i+j] for j in range(len(data)%NB_WINDOWS_BY_SEQ)])
	data = new_data
	x_stages = new_x_stages

	l = [i for i in range(len(data))]
	shuffle(l)
	data_shuffled = [ data[i] for i in l]
	x_stages_shuffled = [ x_stages[i] for i in l]
	test_seqs = data_shuffled[:int(fraction_test*len(data_shuffled))]
	training_seqs = data_shuffled[int(fraction_test*len(data_shuffled)):]

	training_set = setFromList(training_seqs)
	test_set = setFromList(test_seqs)

	saveSet(training_set,"training_set"+name+".txt")
	saveSet(test_set,"test_set"+name+".txt")
	return (data_shuffled,x_stages_shuffled,stages)


n_bins = 6 #nb of letters
nb_states = 10
n_coefs= 4
data,x_stages,stages = write_training_test_set(0.0,n_bins=n_bins,n_coefs=n_coefs)

tr = loadSet("training_set.txt")

alphabet = [''.join(j) for j in list(product(*[[chr(i) for i in range(97,97+n_bins)]]*n_coefs))]

rm = modelHMM_random(nb_states,alphabet,random_initial_state=True)
#rm.save("init_model.txt")
#rm = loadHMM("init_model.txt")

algo = BW_HMM(rm)
out = algo.learn(tr,verbose=True)
out.save("output_model.txt")
out.pprint()
out = loadHMM("output_model.txt")
#print("Loglikelihood on test_set for initial model ",rm.logLikelihood(ts))
#print("Loglikelihood on test_set for output  model ",out.logLikelihood(ts))

algo = BW_HMM(out)
stages_states = []

tot_stages = [0 for i in stages]
for seq in x_stages:
	for i in stages:
		tot_stages[stages.index(i)] +=  seq.count(i)

for i in range(len(stages)):
	stages_states.append([0.0 for i in range(nb_states)])

for i in range(len(data)):
	alphas = algo.computeAlphas(data[i])
	betas  = algo.computeBetas (data[i])
	proba = sum([alphas[s][-1] for s in range(nb_states)])
	for j in range(len(x_stages[i])):
		stage_index = stages.index(x_stages[i][j])
		for s in range(nb_states):
			stages_states[stage_index][s] += (100*alphas[s][j]*betas[s][j])/(proba*tot_stages[stage_index])

for i in range(len(stages)):
	for j in range(nb_states):
		stages_states[i][j] = round(stages_states[i][j],2)
states = ['s'+str(i) for i in range(1,nb_states+1)]

fig, ax = plt.subplots()
im = ax.imshow(stages_states)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(states)), labels=states)	
ax.set_yticks(np.arange(len(stages)), labels=stages)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
		 rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(stages)):
	for j in range(len(states)):
		text = ax.text(j, i, stages_states[i][j],
					   ha="center", va="center", color="w")

ax.set_title("Correlation between stages and states")
fig.tight_layout()
plt.show()