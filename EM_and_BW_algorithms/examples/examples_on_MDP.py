import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from Estimation_algorithms_MDP import *
from examples_models import modelMDP1_fullyobservable, modelMDP3, modelMDP_random, scheduler_random
from MDP import *


alphabet = ['0','1']
actions = ['a','b']
m = modelMDP3()
s = scheduler_random(actions)

training_set_seq = []
training_set_val = []

for i in range(10):
	trace = m.run(5, s, False)

	if not trace in training_set_seq:
		training_set_seq.append(trace)
		training_set_val.append(0)

	training_set_val[training_set_seq.index(trace)] += 1

training_set = [training_set_seq,training_set_val]
m = modelMDP_random(3,alphabet,actions)
m.pprint()
#algo = Estimation_algorithm_MDP_sequences(m, alphabet, actions)
#print(algo.problem3(training_set))

algo = Estimation_algorithm_MDP_schedulers(m, alphabet, actions)
print(algo.problem3([(s,training_set)]))
