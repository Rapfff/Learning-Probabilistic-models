from signal import signal
from pyeeg import *
from edfreader import EDFreader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# transformation : pfd, dfa, hurst
MANUAL_SCORING_WINDOW_SEC = 30
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

psgs = [1]
signal_id = 20
for psg_number in psgs:
	edf_file , hypno_file = file_paths_from_psg_number(psg_number)

	r = EDFreader(edf_file)
	length = r.getTotalSamples(signal_id)
	frequency = r.getSampleFrequency(signal_id)
	
	start_time_edf   = [r.getStartTimeHour(),r.getStartTimeMinute(),r.getStartTimeSecond()]
	h = pd.read_excel(hypno_file)
	start = h["Start Time"][1]
	end   = h["End Time"].values[-1]
	duration   = (end-start).total_seconds()
	diff = (start.hour - start_time_edf[0])*60*60 + (start.minute - start_time_edf[1])*60 + (start.second - start_time_edf[2])
	begining = int(diff*frequency)


	r.fseek(signal_id,begining,EDFreader.EDFSEEK_SET)
	stages = ["Wake","N1","N2","N3","REM"]
	data = {"Wake":[[],[],[]],
			"N1":  [[],[],[]],
			"N2":  [[],[],[]],
			"N3":  [[],[],[]],
			"REM": [[],[],[]]}
	c = 1
	while c*MANUAL_SCORING_WINDOW_SEC <= duration:
		print(c)
		s = h["Event"][c]
		if s in stages:
			vals = read_EDF_signal(r,int(MANUAL_SCORING_WINDOW_SEC*frequency),signal_id)
			data[s][0].append(pfd(vals))
			data[s][2].append(hurst(vals))
			data[s][1].append(dfa(vals))
		c += 1

for s in stages:
	fig, axs = plt.subplots(1, 3)
	axs[0].hist(data[s][0], label='pfd')
	axs[1].hist(data[s][1], label='dfa')
	axs[2].hist(data[s][2], label='vals')
	plt.show()	
