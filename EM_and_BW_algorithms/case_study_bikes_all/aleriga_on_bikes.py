import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from alergia import Alergia
import datetime

f = open("../datasets/bikes/days377_group.csv",'r')
l = f.readline()
l = f.readline()[:-1].split(',')

training_set = [[],[]]

day = 2 # we start with wednesday

while datetime.datetime.strptime(l[0],'%Y-%m-%d') < datetime.datetime(2017,1,1,0,0):
	l = f.readline()[:-1].split(',')
	day = (day+1) % 7

while datetime.datetime.strptime(l[0],'%Y-%m-%d') < datetime.datetime(2019,1,1,0,0):
	if day < 5: # we remove the weekends
		l = l[9:-4] #  we keep just 8am-8pm
		if not l in training_set[0]:
			training_set[0].append(l)
			training_set[1].append(1)
		else:
			training_set[1][training_set[0].index(l)] += 1
	
	l = f.readline()[:-1].split(',')
	day = (day+1) % 7

f.close()
alphabet = ['--5']
alphabet += [ str(i) for i in range(-4,10)]
alphabet.append('++10')

algo = Alergia(training_set,0.05,alphabet)

algo.learn().save("results/modelAlergia.txt")
