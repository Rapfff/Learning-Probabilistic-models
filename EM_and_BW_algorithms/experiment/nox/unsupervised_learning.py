import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

from experiment.nox.edfreader import EDFreader
from src.tools import saveSet, loadSet, setFromList
from src.learning.BW_HMM import BW_HMM
from src.models.HMM import loadHMM
from examples.examples_models import modelHMM_random
import numpy as np
from pyts.approximation import SymbolicFourierApproximation
from itertools import product
from random import shuffle
import pandas as pd

SIGNAL_ID = 44
SIGNAL_NAME = "F3_M2"

WINDOW_SIZE_SEC = 10 #nb of sec as input to DFA
NB_WINDOWS_BY_SEQ = 6 #nb of sec by sequence = WINDOW_SIZE_SEC_MAX*NB_WINDOWS_BY_SEQ

NB_STATES = 5

#PSG 21 BROKEN ??


def file_paths_from_psg_number(nb):
	if nb < 10:
		nb = '0'+str(nb)
	else:
		nb = str(nb)
	edf_file = "/datasets/10x50_psg/edf_recordings/psg"+nb+"/edf_data_export.edf"
	hypno_file = "/datasets/10x50_psg/raw_event_exports/01/psg"+nb+"/xls_events.xls"
	return [edf_file, hypno_file]

def read_EDF_signal(r,size):
	window = np.arange(size,dtype=np.float_)
	r.readSamples(SIGNAL_ID,window,WINDOW_SIZE_SEC)
	return [i for i in window]

def find_starting_point(r,frequency,hypno_file):
	start_time_edf   = [r.getStartTimeHour(),r.getStartTimeMinute(),r.getStartTimeSecond()]
	
	h = pd.read_excel(hypno_file)
	h = h["Start Time"][1]
	start_time_event = [h.hour,h.minute,h.second]
	
	diff = (start_time_event[0] - start_time_edf[0])*60*60 + (start_time_event[1] - start_time_edf[1])*60 + (start_time_event[2] - start_time_edf[2])
	begining = int(diff*frequency)
	return begining

def read_files(psg_number: int):
	edf_file , hypno_file = file_paths_from_psg_number(psg_number)
	
	r = EDFreader(edf_file)
	length = r.getTotalSamples(SIGNAL_ID)
	frequency = r.getSampleFrequency(SIGNAL_ID)
	exp_duration = length/frequency #in seconds

	begining = find_starting_point(r,frequency,hypno_file)
	
	r.fseek(SIGNAL_ID,begining,EDFreader.EDFSEEK_SET)
	data = []
	c = 0
	while (c+1)*WINDOW_SIZE_SEC < exp_duration:
		data.append(read_EDF_signal(r,int(WINDOW_SIZE_SEC*frequency)))
		c += 1
	
	data.append(read_EDF_signal(r,int((exp_duration-c*WINDOW_SIZE_SEC)*frequency)))
	return data


def write_training_test_set(psg_numbers: list,fraction_test: float,name='',n_coefs=4,n_bins=6):
	"""name is the name of the output files,
	fraction_test is a float between ]0,1[ corresponding to the fraction of sequences in the test set """
	if name != '':
		name = '_'+str(name)
	
	new_data = []
	
	for psg_number in psg_numbers:
		print("PSG number:",psg_number)
		data = read_files(psg_number)
		transformer = SymbolicFourierApproximation(n_coefs=n_coefs,n_bins=n_bins)
		data = transformer.fit_transform(data)

		data = [''.join(i) for i in data]
		for i in range(0,len(data) - NB_WINDOWS_BY_SEQ,NB_WINDOWS_BY_SEQ):
			new_data.append([data[i+j] for j in range(NB_WINDOWS_BY_SEQ)])
		new_data.append([data[i+j] for j in range(len(data)%NB_WINDOWS_BY_SEQ)])
	
	data = new_data

	shuffle(data)
	test_seqs = data[:int(fraction_test*len(data))]
	training_seqs = data[int(fraction_test*len(data)):]

	training_set = setFromList(training_seqs)
	saveSet(training_set,"training_set"+name+".txt")
	if fraction_test > 0.0:
		test_set = setFromList(test_seqs)
		#saveSet(test_set,"test_set"+name+".txt")
	return training_set


n_bins = 5 #nb of letters
n_coefs= 4

# size alphabet = n_coefs*n_bins
# nb possible sequences = (size alphabet)**NB_WINDOWS_BY_SEQ 

training_psgs = list(range(1,41))
training_psgs.remove(21)

tr = write_training_test_set(list(range(1,41)),0.0,n_bins=n_bins,n_coefs=n_coefs)
print("number of traces:",sum(tr[1]))
#tr = loadSet("training_set.txt")

alphabet = [''.join(j) for j in list(product(*[[chr(i) for i in range(97,97+n_bins)]]*n_coefs))]

rm = modelHMM_random(NB_STATES,alphabet,random_initial_state=True)
#rm.save("init_model.txt")
#rm = loadHMM("init_model.txt")

algo = BW_HMM(rm)
out = algo.learn(tr, verbose=True)
out.pprint()
out.save("output_model.txt")
#print("Loglikelihood on test_set for initial model ",rm.logLikelihood(ts))
#print("Loglikelihood on test_set for output  model ",out.logLikelihood(ts))
