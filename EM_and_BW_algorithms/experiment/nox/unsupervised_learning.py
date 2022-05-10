import os, sys
from winreg import REG_NO_LAZY_FLUSH

from regex import R
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

from experiment.nox.edfreader import EDFreader
from src.tools import saveSet, setFromList
from src.learning.BW_GOHMM import BW_GOHMM
from src.learning.BW import BW
from src.models.GOHMM import GOHMM
from examples.examples_models import modelGOHMM_nox
import numpy as np
import pandas as pd
from datetime import datetime
from statistics import mean, stdev
from statsmodels.regression.linear_model import yule_walker
MANUAL_SCORING_WINDOW_SEC = 30

WINDOW_SIZE_SEC = 3 
NB_WINDOWS_BY_SEQ = MANUAL_SCORING_WINDOW_SEC//WINDOW_SIZE_SEC

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
	return window

def find_starting_ending_point(r,frequency,hypno_file):
	start_time_edf   = [r.getStartTimeHour(),r.getStartTimeMinute(),r.getStartTimeSecond()]
	
	h = pd.read_excel(hypno_file)
	start = h["Start Time"][1]
	end   = h["End Time"].values[-1]
	duration   = (end-start).total_seconds()
	
	diff = (start.hour - start_time_edf[0])*60*60 + (start.minute - start_time_edf[1])*60 + (start.second - start_time_edf[2])
	begining = int(diff*frequency)
	return (begining,duration)

def read_files(psg_number: int, signal_id: int):
	edf_file , hypno_file = file_paths_from_psg_number(psg_number)
	
	r = EDFreader(edf_file)
	frequency = r.getSampleFrequency(signal_id)
	print(frequency)

	begining, duration = find_starting_ending_point(r,frequency,hypno_file)
	
	r.fseek(signal_id,begining,EDFreader.EDFSEEK_SET)
	data = []
	c = 0
	while (c+1)*WINDOW_SIZE_SEC < duration:
		data.append(read_EDF_signal(r,int(WINDOW_SIZE_SEC*frequency),signal_id))
		c += 1
		if data[-1][0] == 0.0 and data[-1][1] == 1.0:
			print(psg_number,c*WINDOW_SIZE_SEC,duration)
	data.append(read_EDF_signal(r,int((duration-c*WINDOW_SIZE_SEC)*frequency),signal_id))
	return data


def write_set(psg_numbers: list,signal_id,name):
	"""name is the name of the output files,
	fraction_test is a float between ]0,1[ corresponding to the fraction of
	sequences in the test set """
	new_data = []
	for psg_number in psg_numbers:
		print("PSG:",psg_number, "Signal:",signal_id)
		data = read_files(psg_number,signal_id)
		for i,j in enumerate(data):
			rho, _ = yule_walker(j, 2, method='mle')
			data[i] = rho[0]
		for i in range(0,len(data) - NB_WINDOWS_BY_SEQ,NB_WINDOWS_BY_SEQ):
			new_data.append([data[i+j] for j in range(NB_WINDOWS_BY_SEQ)])
		new_data.append([data[i+j] for j in range(len(data)%NB_WINDOWS_BY_SEQ)])

		sleep_stages = ["Wake","N1","N2","N3","REM"]
		xx = [[] for s in sleep_stages]
		h = pd.read_excel(file_paths_from_psg_number(psg_number)[1])
		h = list(h["Event"])[1:]
		for h_i, d_i in zip(h,new_data):
			if h_i in sleep_stages:
				xx[sleep_stages.index(h_i)] += d_i
		for i in range(5):
			print(sleep_stages[i])
			print("Mean:",mean(xx[i]))
			print("Std:", stdev(xx[i]))
		input()

	data = setFromList(new_data)
	if name != False:
		saveSet(data, name+".txt")
	return data

def evaluation(m: GOHMM, signal_id, psg_numbers: list) -> list:
	sleep_stages = ["Wake","N1","N2","N3","REM"]
	bw = BW(m)
	corr_matrix = []
	for i in m.states:
		corr_matrix.append([0 for j in sleep_stages])

	for psg_number in psg_numbers:
		h = pd.read_excel(file_paths_from_psg_number(psg_number)[1])
		h = list(h["Event"])[1:]
		g = write_set([psg_number],signal_id,"test_set")

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



signals_ids  = [     20] #,     24,     30,    34,      44,     48,     67,     71] 
signals_names = ["C3-M2"] #,"C4-M1","E1-M2","E2-M1","F3-M2","F4-M1","O1-M2","O2-M1"]
psgs = list(range(1,51))
#shuffle(psgs)
training_psgs = [psgs[0]]
#test_psgs = psgs[25:30]
running_times = []

for signal_id, signal_name in zip(signals_ids, signals_names):
	tr = write_set(training_psgs,signal_id,"training_set")
	rm = modelGOHMM_nox(NB_STATES,random_initial_state=True)
	algo = BW_GOHMM(rm)
	starting_time = datetime.now()
	out = algo.learn(tr, output_file="model_"+signal_name+".txt")
	running_times.append((datetime.now()-starting_time).total_seconds())
	"""
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
	print(string)
	"""
out.pprint()
print("Average learning time",mean(running_times))
