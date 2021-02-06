import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from Estimation_algorithms_MDP import *
from examples_models import modelMDP1_fullyobservable, modelMDP3, modelMDP_random, scheduler_random
from MDP import *

def fullyobservable_sequences(mdp, scheduler, length_sequences, nb_sequences):
	print("Real MDP:")
	mdp.pprint()
	print()
	sequences = []
	for i in range(nb_sequences):
		sequences.append(mdp.run(length_sequences, scheduler))
	print("Learned MDP:")
	Estimation_algorithm_fullyobservable_MDP().learnFromSequences(sequences).pprint()

def fullyobservable_blackbox(mdp, length_sequences, nb_sequences):
	print("Real MDP:")
	mdp.pprint()
	print()
	print("Learned MDP:")
	Estimation_algorithm_fullyobservable_MDP().learnFromBlackBox(mdp, nb_sequences,length_sequences).pprint()




alphabet = ['0','1']
actions = ['a','b']
m = modelMDP3()
s = scheduler_random(actions)

training_set_seq = []
training_set_val = []

for i in range(100):
	trace = m.run(5, s)

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
print(algo.problem3([s,training_set]))
