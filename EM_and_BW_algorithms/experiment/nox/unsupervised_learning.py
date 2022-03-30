import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

from experiment.nox.edfreader import EDFreader
from src.tools import saveSet, loadSet, setFromList
from src.learning.BW_HMM import BW_HMM
from src.learning.BW import BW
from src.models.HMM import loadHMM, HMM
from examples.examples_models import modelHMM_random
import numpy as np
from pyts.approximation import SymbolicFourierApproximation
from itertools import product
from random import shuffle
import pandas as pd

SIGNAL_ID = 44
SIGNAL_NAME = "F3_M2"

MANUAL_SCORING_WINDOW_SEC = 30

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
	hypno_file = "/datasets/10x50_psg/raw_event_exports/01/psg"+nb+"/xls_hypnogram.xls"
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


def write_set(psg_numbers: list,name,n_coefs=4,n_bins=6,shuffling=True):
	"""name is the name of the output files,
	fraction_test is a float between ]0,1[ corresponding to the fraction of sequences in the test set """
	
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
	if shuffling:
		shuffle(data)

	data = setFromList(data)

	if name != False:
		saveSet(data, name+".txt")
	return data

def evaluation(m: HMM, psg_numbers: list) -> list:
	sleep_stages = ["Wake","N1","N2","N3","REM"]
	bw = BW(m)
	corr_matrix = []
	for i in m.states:
		corr_matrix.append([0 for j in sleep_stages])

	for psg_number in psg_numbers:
		h = pd.read_excel(file_paths_from_psg_number(psg_number)[1])
		h = list(h["Event"])[1:]
		g = write_set([psg_number],False,shuffling=False)

		for seq in range(len(g[0])):
			alphas = bw.computeAlphas(g[0][seq])
			betas  = bw.computeBetas(g[0][seq])

			for t in range(len(g[0][seq])):
				alphas_betas = [alphas[s][t+1]*betas[s][t+1] for s in range(len(m.states))]
				chosen = alphas_betas.index(max(alphas_betas))
				index_h = int((t+seq*NB_WINDOWS_BY_SEQ)*WINDOW_SIZE_SEC/MANUAL_SCORING_WINDOW_SEC)
				print(h[index_h])
				if h[index_h] in sleep_stages:
					corr_matrix[chosen][sleep_stages.index(h[index_h])] += g[1][seq]
	return corr_matrix



n_bins = 5 #nb of letters
n_coefs= 4

# size alphabet = n_bins**n_coefs
# nb possible sequences = (size alphabet)**NB_WINDOWS_BY_SEQ 

training_psgs = list(range(1,41))
training_psgs.remove(21)
test_psgs = list(range(41,51))

# write_set(training_psgs,"training_set",n_coefs,n_bins)
# write_set(test_psgs,"test_set",n_coefs,n_bins)

# tr = loadSet("training_set.txt")
# ts = loadSet("test_set.txt")
 
# alphabet = [''.join(j) for j in list(product(*[[chr(i) for i in range(97,97+n_bins)]]*n_coefs))]
 
# rm = modelHMM_random(NB_STATES,alphabet,random_initial_state=True)

# algo = BW_HMM(rm)
# out = algo.learn(tr, verbose=True)
# out.pprint()
# out.save("output_model.txt")

out = loadHMM("output_model.txt")
corr_matrix = evaluation(out, test_psgs)

print(" "*8+'|  Wake  |   N1   |   N2   |   N3   |  REM   ',end='')
for i in range(len(corr_matrix)):
	row = corr_matrix[i]
	print('\n'+'-'*53)
	print("   s"+str(i)+"   ",end="")
	for j in row:
		s = str(j)
		print('|'+" "*(8-len(s))+s,end="")
print('\n'+'-'*53)

