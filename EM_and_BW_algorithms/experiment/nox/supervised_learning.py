import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

from experiment.nox.edfreader import EDFreader
from src.tools import saveSet, loadSet
from src.learning.BW_GOHMM import BW_GOHMM
from src.learning.BW import BW
from src.models.GOHMM import GOHMM
from examples.examples_models import modelGOHMM_random
import numpy as np
import pandas as pd
from statistics import mean, stdev
MANUAL_SCORING_WINDOW_SEC = 30
AUTOMATIC_SCORING_WINDOW_SEC = 1

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
	begining, duration = find_starting_ending_point(r,frequency,hypno_file)
	
	r.fseek(signal_id,begining,EDFreader.EDFSEEK_SET)
	data = []
	c = 0
	while (c+1)*AUTOMATIC_SCORING_WINDOW_SEC < duration:
		data.append(read_EDF_signal(r,int(AUTOMATIC_SCORING_WINDOW_SEC*frequency),signal_id))
		c += 1
	return data

def normalize(ll):
	ll = np.array(ll)
	m = np.mean(ll)
	s = np.std(ll)
	return (ll-m)/s


def write_set(psg_numbers: list,signal_id,name):
	"""name is the name of the output files,
	fraction_test is a float between ]0,1[ corresponding to the fraction of
	sequences in the test set """
	ts = [ [[],[]] for s in sleep_stages]
	for psg_number in psg_numbers:
		print("PSG:",psg_number, "Signal:",signal_id)
		data = read_files(psg_number,signal_id)
		data = normalize(data)
		h = pd.read_excel(file_paths_from_psg_number(psg_number)[1])
		h = list(h["Event"])[1:]
		for k in range(len(data)):
			manual = h[(k*AUTOMATIC_SCORING_WINDOW_SEC)//MANUAL_SCORING_WINDOW_SEC]
			if manual in sleep_stages:
				ts[sleep_stages.index(manual)][0].append(data[k])
				ts[sleep_stages.index(manual)][1].append(1)
	for i,s in enumerate(sleep_stages):
		saveSet(ts[i],s+'_'+name+".txt")
	return ts

def string_correlation_matrix(corr_matrix):
	string = " "*8+'|  Wake  |   N1   |   N2   |   N3   |  REM    '
	for i in range(len(corr_matrix)):
		row = corr_matrix[i]
		string += '\n'+'-'*53+'\n'
		string += "   s"+str(i)+"   "
		for j in row:
			s = str(j)
			string += '|'+" "*(8-len(s))+s
	string += '\n'+'-'*53+'\n'
	return string

signal_id = 20
signal_name = "C3-M2"
training_psg = [1,2,3]
test_psg = [4,5]
sleep_stages = ["Wake","N1","N2","N3","REM"]

ts = write_set(training_psg,signal_id,"training")
#ts = [loadSet(s+'_training.txt') for s in sleep_stages]
init = [modelGOHMM_random(NB_STATES,True,-1.0,1.0,0.1,2.0) for _ in sleep_stages]
out = []
for i,s in enumerate(sleep_stages):
	print("Learning:",s)
	bw = BW_GOHMM(init[i])
	out.append(bw.learn(ts[i],s+"_model.txt"))

ts = write_set(test_psg,signal_id,"test")
corr = [[0 for s in sleep_stages] for i in out]
for i,s in enumerate(sleep_stages):
	print("Testing:",s)
	for seq in ts[i][0]:
		ll = [m.logLikelihood(seq) for m in out]
		corr[ll.index(max(ll))][i] += 1

print(string_correlation_matrix(corr))


