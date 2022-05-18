import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

from experiment.nox.edfreader import EDFreader
from src.tools import saveSet, loadSet
from src.learning.BW_GOHMM import BW_GOHMM
from src.learning.BW import BW
from src.models.GOHMM import GOHMM, loadGOHMM
from examples.examples_models import modelGOHMM_nox
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.integrate import simps
from statistics import mean, stdev
MANUAL_SCORING_WINDOW_SEC = 30
AUTOMATIC_SCORING_WINDOW_SEC = 1
SEQUENCE_SIZE = 90
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
		v = read_EDF_signal(r,int(AUTOMATIC_SCORING_WINDOW_SEC*frequency),signal_id)
		data.append(bandpower(v,frequency,[0.5,4],5,True))
		c += 1
	return data

def bandpower(data, sf, band, window_sec=None, relative=False):
    low, high = band
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf
    freqs, psd = welch(data, sf, nperseg=nperseg)
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    bp = simps(psd[idx_band], dx=freq_res)
    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp

def splitInSequences(ll):
	res = [[],[]]
	for i in range(len(ll)//SEQUENCE_SIZE):
		seq = ll[i*SEQUENCE_SIZE:(i+1)*SEQUENCE_SIZE]
		if not seq in res[0]:
			res[0].append(seq)
			res[1].append(1)
		else:
			res[1][res[0].index(seq)] += 1
	res[0].append(ll[(i+1)*SEQUENCE_SIZE:])
	res[1].append(1)
	return res

def write_set(psg_numbers: list,signal_id,name=None):
	for psg_number in psg_numbers:
		print("PSG:",psg_number, "Signal:",signal_id)
		data = read_files(psg_number,signal_id)
		data = splitInSequences(data)
	if name:
		saveSet(data,name+".txt")
	return data

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

def evaluation(psg_numbers,m):
	corr = [[0 for s in sleep_stages] for i in range(NB_STATES)]
	bw = BW(m)
	for psg_number in psg_numbers:
		h = pd.read_excel(file_paths_from_psg_number(psg_number)[1])
		h = list(h["Event"])[1:]
		g = write_set([psg_number],signal_id)

		for seq in range(len(g[0])):
			alphas = bw.computeAlphas(g[0][seq])
			betas  = bw.computeBetas(g[0][seq])
			for t in range(len(g[0][seq])):
				alphas_betas = [alphas[s][t]*betas[s][t] for s in range(len(m.states))]
				index_h = int(t+seq*SEQUENCE_SIZE)
				if index_h >= len(h):
					break
				if h[index_h] in sleep_stages:
					chosen = alphas_betas.index(max(alphas_betas))
					corr[chosen][sleep_stages.index(h[index_h])] += g[1][seq]
					#for s in range(len(m.states)):
					#	corr[s][sleep_stages.index(h[index_h])] += g[1][seq]*alphas_betas[s]
	return corr

signal_id = 44
signal_name = "F3-M2"
training_psg = [1,2,3]
test_psg = [4,5]
sleep_stages = ["Wake","N1","N2","N3","REM"]

ts = write_set(training_psg,signal_id,"training")
#ts = loadSet('training.txt')
init = modelGOHMM_nox()
bw = BW_GOHMM(init)
out = bw.learn(ts,'output_model.txt')

#ts = [loadSet(s+'_test.txt') for s in sleep_stages]
out = loadGOHMM("output_model.txt")
print("Testing:")
corr = evaluation(test_psg,out)

print(string_correlation_matrix(corr))


