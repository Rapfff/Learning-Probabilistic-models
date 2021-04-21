import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from examples.examples_models import modelMDP_random, scheduler_uniform
from examples.small_gridMDP import SmallGrid
from Estimation_algorithms_MDP_multithreading import Estimation_algorithm_MDP
from datetime import datetime
from tools import randomProbabilities, generateSet
from Active_Learning_MDP import Active_Learning_MDP
from MDP import loadMDP

def mergeSets(s1,s2):
	for seq in s2[0]:
		if seq in s1[0]:
			s1[1][s1[0].index(seq)] += 1
		else:
			s1[0].append(seq)
			s1[1].append(1)
	return s1

nb_states = 9
original = SmallGrid()
observations = original.observations()
actions = original.actions()

set_size_learning = 100
set_size_test = 500
sequence_size = 10

s = scheduler_uniform(actions)
"""
training_set = generateSet(m,set_size_learning,sequence_size,s,True)
test_set = generateSet(m,set_size_learning,sequence_size,s,True)

f = open("training_set.txt",'w')
for i in range(len(training_set[0])):
	f.write(str(training_set[0][i])+'\n')
	f.write(str(training_set[1][i])+'\n')
f.close()
f = open("test_set.txt",'w')
for i in range(len(test_set[0])):
	f.write(str(test_set[0][i])+'\n')
	f.write(str(test_set[1][i])+'\n')
f.close()
m = modelMDP_random(nb_states,observations,actions)
m.save("initial_model.txt")
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


# -----------------------------------------------
m = loadMDP("initial_model.txt")
print(datetime.now(),"Active Learning:")
algo = Active_Learning_MDP(m,observations,actions)
algo.learn(training_set,0.1,0.25,20,50,limit=0.01)
history = algo.history
# -----------------------------------------------
m = loadMDP("initial_model.txt")
print(datetime.now(),"Passive Learning:")
algo = Estimation_algorithms_MDP_multithreading(m,observations,actions)
algo.problem3(training_set,"passive_models/passive_models_0.txt",0.01,'0')
for i in range(len(history)):
	m = algo.h
	algo = Estimation_algorithms_MDP_multithreading(m,observations,actions)
	t = generateSet(original,history[i],sequence_size,s,True)
	algo.problem3(mergeSets(training_set,t),"passive_models/passive_models_"+str(i+1)+".txt",0.01,str(i+1)+'/'+str(len(history)+1))

"""
import matplotlib.pyplot as plt
test_set = [[],[]]
f = open("test_set.txt",'r')
l = f.readline()
while l:
	l = l.replace("'",'')
	l = l.replace(' ','')
	test_set[0].append(l[1:-2].split(','))
	l = f.readline()
	test_set[1].append(int(l[:-1]))
	l = f.readline()
f.close()

results_active = []
results_passive= []
for c in range(len(history)+1):
	m = loadMDP("active_models/active_models_"+str(c)+".txt")
	results_active.append(m.logLikelihoodTraces(test_set))
	m = loadMDP("passive_models/passive_models_"+str(c)+".txt")
	results_passive.append(m.logLikelihoodTraces(test_set))

x_val = [set_size_learning+sum(history[:i+1]) for i in range(len(history))]
x_val.insert(0,set_size_learning)

x = [i for i in range(len(x_val))]
fig, ax = plt.subplots()
rects1 = ax.bar([i - width/2 for i in x], results_active, width, label='Active Learning')
rects2 = ax.bar([i + width/2 for i in x], results_passive, width, label='Passive Learning')

width = 0.35  # the width of the bars
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('lolikelihood of the test set')
ax.set_ylabel('size of the test set')
ax.set_title('Active Learning vs Passive Learning')
ax.set_xticks(x)
ax.set_xticklabels([str(i) for i in x_val])
ax.legend()

ax.bar_label(results_active, padding=3)
ax.bar_label(results_passive, padding=3)

fig.tight_layout()

plt.show()
"""