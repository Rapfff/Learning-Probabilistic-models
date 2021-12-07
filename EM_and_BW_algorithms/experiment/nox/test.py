from edfreader import EDFreader
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from scipy.signal import hilbert
from math import exp

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
		x_stages[s].append([c+i for i in range(-1,d,1)])
		l = events.readline()
		c += d
	events.close()
	return [hil,stages,x_stages]

def write_training_sets(hil,stages,x_stages):
	for s in stages:
		f = open(str(s)+"_trainingset.txt",'w')
		for j in x_stages[s]:
			for i in j:
				f.write(str(hil[i])+',')
		f.close()

def discretize(hil):
	m1 = min(hil)
	m2 = max(hil)-m1
	hil = [100*(i-m1)/m2 for i in hil]
	hil_sorted = hil[:]
	hil_sorted.sort()

	for i in range(len(hil)):
		for j in range(1,SIZE_ALPHABET+1):
			if hil[i] <= hil_sorted[int((j*len(hil_sorted)/SIZE_ALPHABET)-1)]:
				hil[i] = j
				break
	return hil

hil, stages, x_stages = read_files()
hil = discretize(hil)
write_training_sets(hil,stages,x_stages)




#IDEE:
#WINDOW_SIZE_SEC = 1
#pour chaque sec => 1 hilbert value
#séparer le training set pour chaque stage
#train un model par stage sur son training set
#pour sequence de 30 secondes (30 hilbert values), calculer la proba que chaque model génere cette  sequence


"""
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
"""


#plt.plot(range(window_size),window)
#plt.plot(range(window_size),[sum(window)/window_size for i in range(window_size)])
#plt.show()
#print(peak_to_peak_amplitude_and_freq(window))