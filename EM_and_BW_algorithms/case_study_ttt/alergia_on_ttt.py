import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from IOalergia import IOAlergia

training_set = [[],[]]



f = open("training_set.txt",'r')
l = f.readline()
while l:
	l = l[1:-2]
	l = l.replace(' ','')
	l = l.replace("'",'')
	training_set[0].append(l.split(','))
	l = f.readline()
	training_set[1].append(int(l[:-1]))
	l = f.readline()

f.close()

actions = ['1','2','3','4','5','6','7','8','9']
observations = ['1','2','3','4','5','6','7','8','9',"error","win","loose","draw"]

algo = IOAlergia(training_set,0.5,actions,observations)

algo.learn().pprint()
#algo.learn().save("results/modelAlergia.txt")
