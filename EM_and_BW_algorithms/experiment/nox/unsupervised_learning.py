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
from random import shuffle
import pandas as pd
from datetime import datetime
from statistics import mean, stdev

MANUAL_SCORING_WINDOW_SEC = 30

WINDOW_SIZE_SEC = 30 #nb of sec as input to DFA
NB_WINDOWS_BY_SEQ = 180 #nb of sec by sequence = WINDOW_SIZE_SEC_MAX*NB_WINDOWS_BY_SEQ

NB_STATES = 5

#PSG 21 BROKEN ??


def file_paths_from_psg_number(nb):
	if nb < 10:
		nb = '0'+str(nb)
	else:
		nb = str(nb)
	edf_file = "/datasets/10x100/psg/edf_recordings/psg"+nb+"/edf_data_export.edf"
	hypno_file = "/datasets/10x100/psg/raw_event_exports/01/psg"+nb+"/xls_hypnogram.xls"
	return [edf_file, hypno_file]

def read_EDF_signal(r,size,signal_id):
	window = np.arange(size,dtype=np.float_)
	r.readSamples(signal_id,window,size)
	return [i for i in window]

def find_starting_point(r,frequency,hypno_file):
	start_time_edf   = [r.getStartTimeHour(),r.getStartTimeMinute(),r.getStartTimeSecond()]
	
	h = pd.read_excel(hypno_file)
	h = h["Start Time"][1]
	start_time_event = [h.hour,h.minute,h.second]
	
	diff = (start_time_event[0] - start_time_edf[0])*60*60 + (start_time_event[1] - start_time_edf[1])*60 + (start_time_event[2] - start_time_edf[2])
	begining = int(diff*frequency)
	return begining

def read_files(psg_number: int, signal_id: int):
	edf_file , hypno_file = file_paths_from_psg_number(psg_number)
	
	r = EDFreader(edf_file)
	length = r.getTotalSamples(signal_id)
	frequency = r.getSampleFrequency(signal_id)
	exp_duration = length/frequency #in seconds

	begining = find_starting_point(r,frequency,hypno_file)
	
	r.fseek(signal_id,begining,EDFreader.EDFSEEK_SET)
	data = []
	c = 0
	while (c+1)*WINDOW_SIZE_SEC < exp_duration:
		data.append(read_EDF_signal(r,int(WINDOW_SIZE_SEC*frequency),signal_id))
		c += 1
	
	data.append(read_EDF_signal(r,int((exp_duration-c*WINDOW_SIZE_SEC)*frequency),signal_id))
	return data


def write_set(psg_numbers: list,signal_id,name,n_coefs=4,n_bins=6):
	"""name is the name of the output files,
	fraction_test is a float between ]0,1[ corresponding to the fraction of sequences in the test set """
	
	new_data = []
	
	for psg_number in psg_numbers:
		print("PSG number:",psg_number)
		data = read_files(psg_number,signal_id)
		for x in range(len(data)):
			for y in range(len(data[x])):
				if type(data[x][y]) == list:
					print(x,y,data[x][y],type(data[x][y]))
		transformer = SymbolicFourierApproximation(n_coefs=n_coefs,n_bins=n_bins)
		data = transformer.fit_transform(data)
		print(len(data))
		print(len(data[0]))

		data = [''.join(i) for i in data]
		for i in range(0,len(data) - NB_WINDOWS_BY_SEQ,NB_WINDOWS_BY_SEQ):
			new_data.append([data[i+j] for j in range(NB_WINDOWS_BY_SEQ)])
		new_data.append([data[i+j] for j in range(len(data)%NB_WINDOWS_BY_SEQ)])
	
	data = new_data

	data = setFromList(data)

	if name != False:
		saveSet(data, name+".txt")
	return data

def evaluation(m: HMM, signal_id, psg_numbers: list) -> list:
	sleep_stages = ["Wake","N1","N2","N3","REM"]
	bw = BW(m)
	corr_matrix = []
	for i in m.states:
		corr_matrix.append([0 for j in sleep_stages])

	for psg_number in psg_numbers:
		h = pd.read_excel(file_paths_from_psg_number(psg_number)[1])
		h = list(h["Event"])[1:]
		g = write_set([psg_number],signal_id,False)

		for seq in range(len(g[0])):
			alphas = bw.computeAlphas(g[0][seq])
			betas  = bw.computeBetas(g[0][seq])

			for t in range(len(g[0][seq])):
				alphas_betas = [alphas[s][t+1]*betas[s][t+1] for s in range(len(m.states))]
				index_h = int((t+seq*NB_WINDOWS_BY_SEQ)*WINDOW_SIZE_SEC/MANUAL_SCORING_WINDOW_SEC)
				if index_h >= len(h):
					break
				if h[index_h] in sleep_stages:
					chosen = alphas_betas.index(max(alphas_betas))
					corr_matrix[chosen][sleep_stages.index(h[index_h])] += g[1][seq]
					#for s in range(len(m.states)):
					#	corr_matrix[s][sleep_stages.index(h[index_h])] += g[1][seq]*alphas_betas[s]
	return corr_matrix



n_bins  = 5 # nb of letters
alphabet = list("abcdefghijklmnopqrstuvwxyz")[:n_bins]
n_coefs = 5 # 5 because delta, theta, alpha, beta1, beta2 activity

#list_signals = [     20,     24,     30,    34,      44,     48,     67,     71] 
#signals_name = ["C3-M2","C4-M1","E1-M2","E2-M1","F3-M2","F4-M1","O1-M2","O2-M1"]
list_signals = [44] 
signals_name = ["F3-M2"]


# size alphabet = n_bins**n_coefs
# nb possible sequences = (size alphabet)**NB_WINDOWS_BY_SEQ 
psgs = list(range(1,51))
psgs.remove(21)
shuffle(psgs)
training_psgs = psgs[:44]
test_psgs = psgs[44:]

running_times = []

for signal_index in range(len(list_signals)):
	signal_id = list_signals[signal_index]
	signal_name = signals_name[signal_index]
	
	write_set(training_psgs,signal_id,"training_set",n_coefs,n_bins)
	write_set(test_psgs,signal_id,"test_set",n_coefs,n_bins)
	tr = loadSet("training_set.txt")
	ts = loadSet("test_set.txt")
	
	rm = modelHMM_random(NB_STATES,alphabet,random_initial_state=True)
	algo = BW_HMM(rm)
	starting_time = datetime.now()
	out = algo.learn(tr, output_file="model_"+signal_name, verbose=True)
	running_times.append((datetime.now()-starting_time).total_seconds())
	
	corr_matrix = evaluation(out, signal_id, test_psgs)
	string  = signal_name+'\n'
	string += " "*8+'|  Wake  |   N1   |   N2   |   N3   |  REM   '
	for i in range(len(corr_matrix)):
		row = corr_matrix[i]
		string += '\n'+'-'*53+'\n'
		string += "   s"+str(i)+"   "
		for j in row:
			s = str(j)
			string += '|'+" "*(8-len(s))+s
	string += '\n'+'-'*53+'\n'
	f = open("report_"+signal_name+".txt",'w')
	f.write(string)
	f.close()

print("Average learning time",mean(running_times))
