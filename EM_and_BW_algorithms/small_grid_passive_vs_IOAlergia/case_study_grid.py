import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from examples.examples_models import modelMDP_random, scheduler_uniform
from examples.small_gridMDP import SmallGrid
from Estimation_algorithms_MDP_multithreading import Estimation_algorithm_MDP
from datetime import datetime
from tools import randomProbabilities, generateSet

nb_states = 9
m = SmallGrid()
observations = m.observations()
actions = m.actions()
"""
set_size = 500
sequence_size = 10

s = scheduler_uniform(actions)
training_set = generateSet(m,set_size,sequence_size,s,True)

f = open("training_set.txt",'w')
for i in range(len(training_set[0])):
	f.write(str(training_set[0][i])+'\n')
	f.write(str(training_set[1][i])+'\n')
f.close()
"""
training_set = [[],[]]

f = open("training_set.txt",'r')
l = f.readline()
while l:
	l = l.replace("'",'')
	l = l.replace(' ','')
	training_set[0].append(l[1:-2].split(','))
	l = f.readline()
	training_set[1].append(int(l[:-1]))
	l = f.readline()
f.close()

nb_experiments = 10

for k in range(nb_experiments):
	print(datetime.now(),"iteration:",k)
	m = modelMDP_random(nb_states,observations,actions)
	algo = Estimation_algorithm_MDP(m,observations,actions)
	algo.problem3(training_set,"modelEM_"+str(k+1)+".txt",0.001,k)
