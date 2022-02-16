import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, stdev
from scipy.signal import hilbert
from math import exp
from experiment.nox.edfreader import EDFreader
from examples.examples_models import modelMCGT_random, modelCOHMM_nox
from src.tools import saveSet, loadSet, setFromList
from src.learning.BW_coHMM import BW_coHMM
from random import shuffle


EDF_FILE   = "eh_20210211.edf"
EVENT_FILE = "Event_Grid.csv"
SIZE_ALPHABET = 10
#SIGNALS
#HEART_RATE = 52
#ACTIVITY   = 13
#F3_M2 = 44
SIGNAL_ID = 44
SIGNAL_NAME = "F3_M2"

WINDOW_SIZE_SEC = 1
EVALUATING_WINDOW_SIZE_SEC = 30

NB_STATES = 5

def clean(s):
    i = 1
    while i < len(s):
        if s[i] != s[0]:
            break
        i+=1

    if i/len(s) > 0.02:
        s = s[i:]

    if len(s) > 2:
        i = len(s)-2
        while i >= 0:
            if s[i] != s[-1]:
                break
            i-=1
        if 1.0-i/len(s) > 0.02:
            s = s[:i+1]

    return s

def normalize(seq):
	mu  = mean(seq)
	std = stdev(seq)
	return [(i-mu)/std for i in seq]

def peak_to_peak_amplitude_and_freq(s):
	m = sum(s)/len(s)
	s = [i-m for i in s if i-m != 0]
	peaks_size = []
	up   = []
	down = []
	c = 1
	if s[0] > 0:
		up.append(s[0])
	else:
		down.append(s[0])
	i = 1
	while i < len(s):
		if s[i-1]*s[i] < 0 and len(up)>0 and len(down)>0:
			c +=1
			peaks_size.append(mean(up)-mean(down))
			up = []
			down = []
		if s[i] > 0:
			up.append(s[i])		
		if s[i] < 0:
			down.append(s[i])
		i += 1
	if len(up)>0 and len(down)>0:
		peaks_size.append(mean(up)-mean(down))
	return (mean(peaks_size),c,m)

def read_files():
	events = open(EVENT_FILE,'r')
	l = events.readline()
	l = events.readline()
	l = events.readline()

	r = EDFreader(EDF_FILE)
	length = r.getTotalSamples(SIGNAL_ID)
	frequency = r.getSampleFrequency(SIGNAL_ID)
	exp_duration = length/r.getSampleFrequency(SIGNAL_ID) #in seconds

	start_time_event = [int(i[:2]) for i in l.split(';')[2].split(':')]
	start_time_edf   = [r.getStartTimeHour(),r.getStartTimeMinute(),r.getStartTimeSecond()]
	diff = (start_time_event[0] - start_time_edf[0])*60*60 + (start_time_event[1] - start_time_edf[1])*60 + (start_time_event[2] - start_time_edf[2])
	begining = int(diff*frequency)
	window_size = int(WINDOW_SIZE_SEC*frequency)
	#begining = 4134971
	window = np.arange(window_size,dtype=np.float_)
	r.fseek(SIGNAL_ID,begining,EDFreader.EDFSEEK_SET)

	#amp  = []
	#freq = []
	#means= []
	hil = []

	while r.ftell(SIGNAL_ID)+window_size < length:
		window = np.arange(window_size,dtype=np.float_)
		r.readSamples(SIGNAL_ID,window,window_size)
		#window = clean(window)
		#if len(window)>0:
		#ta,tf,tm = peak_to_peak_amplitude_and_freq(window)
		#amp.append(ta)
		#freq.append(tf)
		#means.append(tm)
		analytic_signal = hilbert(window)
		hil.append(mean(abs(analytic_signal)))

	stages = []
	x_stages = {}
	l = events.readline()
	c = 1
	while l:
		l = l.split(';')
		s = l[0]
		d = int(l[1])//WINDOW_SIZE_SEC
		if not s in stages:
			stages.append(s)
			x_stages[s] = []
		#x_stages[s].append([c+i for i in range(-1,d,1)])
		for i in range(-1,d,1):
			x_stages[s].append(c+i)
		l = events.readline()
		c += d
	events.close()
	return [hil,stages,x_stages]

def naive_analysis():
	hil, stages, x_stages = read_files()
	hil = normalize(hil)

	hil = [(i-mu)/std for i in hil] #normalizing
	mu   = {}
	var  = {}
	trans= {}
	for s in stages:
		mu[s]    = 0.0
		var[s]   = 0.0
		trans[s] = [0.0 for i in range(len(stages))]

	for s in stages:
		if 0 in x_stages[s]:
			break

	for step in range(len(hil)-1):
		mu[s] += hil[step]
		for ss in stages:
			if step+1 in x_stages[ss]:
				break
		trans[s][stages.index(ss)] += 1
		s = ss
	mu[s] += hil[-1]
	for s in stages:
		mu[s] /= len(x_stages[s])
		trans[s] = [j/len(x_stages[s]) for j in trans[s]]

	for step in range(len(hil)-1):
		for s in stages:
			if step+1 in x_stages[s]:
				break
		var[s] += (hil[step]-mu[s])**2
	for s in stages:
		var[s] /= len(x_stages[s])
		print("\nSTAGE",s,"*****************")
		print("Mean    :",round(mu[s],5))
		print("Variance:",round(var[s],5))
		for ss in stages:
			print(s,"=>",ss,":",round(100*trans[s][stages.index(ss)],5),"%")
	
def write_training_test_set(fraction_test,name=''):
	"""name is the name of the output files,
	fraction_test is a float between ]0,1[ correspondin to the fraction of sequences in the test set """
	hil, stages, x_stages = read_files()
	hil = normalize(hil)
	seqs = []
	for i in range(len(hil)//EVALUATING_WINDOW_SIZE_SEC):
		seqs.append(hil[i*EVALUATING_WINDOW_SIZE_SEC:(i+1)*EVALUATING_WINDOW_SIZE_SEC])
	seqs.append(hil[(i+1)*EVALUATING_WINDOW_SIZE_SEC:])
	shuffle(seqs)
	test_seqs = seqs[:int(fraction_test*len(seqs))]
	training_seqs = seqs[int(fraction_test*len(seqs)):]

	training_set = setFromList(training_seqs)
	test_set = setFromList(test_seqs)

	saveSet(training_set,"training_set"+name+".txt")
	saveSet(test_set,"test_set"+name+".txt")
#IDEE:
#WINDOW_SIZE_SEC = 1
#pour chaque sec => 1 hilbert value
#separer le tout en sequences de EVALUATING_WINSOW_SIZE_SEC
#generer un training set avec 1-<fraction_test> des seq et l'inverse pour le training set
#apprendre un model avec autant de states que de stages et une distr sur l'initial state


#write_training_test_set(0.1)
tr = loadSet("training_set.txt",True)
ts = loadSet("test_set.txt",True)

rm = modelCOHMM_nox(self_loop_prob=0.5)
algo = BW_coHMM(rm)
out = algo.learn(tr,verbose=True)
out.pprint()

print("Loglikelihood on test_set for initial model ",rm.logLikelihood(ts))
print("Loglikelihood on test_set for output  model ",out.logLikelihood(ts))


def write_training_sets_each_stage(hil,stages,x_stages):
	for s in stages:
		for j in x_stages[s]:
			seq = []
			t   = []
			for i in range(0,len(j),EVALUATING_WINDOW_SIZE_SEC):
				ss = j[i:min(len(j),i+EVALUATING_WINDOW_SIZE_SEC)]
				ss = [hil[k] for k in ss]
				if ss in seq:
					t[seq.index(ss)] += 1
				else:
					seq.append(ss)
					t.append(1)
		saveSet([seq,t],str(s)+"_training.txt")

#IDEE:
#WINDOW_SIZE_SEC = 1
#pour chaque sec => 1 hilbert value
#séparer le training set pour chaque stage
#train un model par stage sur son training set
#pour sequence de 30 secondes (30 hilbert values), calculer la proba que chaque model génere cette  sequence
"""
write_training_sets(hil,stages,x_stages)

model_stages = []
rm = modelMCGT_random(NB_STATES,stages)
for s in stages:
	model_stages.append(rm)
	algo = Estimation_algorithm_MCGT(model_stages[-1],stages)
	algo.learn(loadSet(str(s)+"_training.txt"))
	model_stages[-1] = algo.learn(loadSet(str(s)+"_training.txt"),output_file=str(s)+"_model.txt",pp=str(s))	

"""



def printing_stuffs(stages,x_stages,amp,freq,means):
	fig, ax = plt.subplots()
	for s in stages:
		first = True
		for x in x_stages[s]:
			if first:
				first = False
				plt.plot(x,[amp[i] for i in x],color=colors[stages.index(s)],label=s)
			else:
				plt.plot(x,[amp[i] for i in x],color=colors[stages.index(s)])
	ax.legend()
	ax.set_title("Amplitude "+SIGNAL_NAME)
	plt.show()

	fig, ax = plt.subplots()
	for s in stages:
		first = True
		for x in x_stages[s]:
			if first:
				first = False
				plt.plot(x,[freq[i] for i in x],color=colors[stages.index(s)],label=s)
			else:
				plt.plot(x,[freq[i] for i in x],color=colors[stages.index(s)])
	ax.legend()
	ax.set_title("Frequency "+SIGNAL_NAME)
	plt.show()


	fig, ax = plt.subplots()
	for s in stages:
		first = True
		for x in x_stages[s]:
			if first:
				first = False
				plt.plot(x,[means[i] for i in x],color=colors[stages.index(s)],label=s)
			else:
				plt.plot(x,[means[i] for i in x],color=colors[stages.index(s)])
	ax.legend()
	ax.set_title("Mean "+SIGNAL_NAME)
	plt.show()
	#plt.plot(range(window_size),window)
	#plt.plot(range(window_size),[sum(window)/window_size for i in range(window_size)])
	#plt.show()
	#print(peak_to_peak_amplitude_and_freq(window))