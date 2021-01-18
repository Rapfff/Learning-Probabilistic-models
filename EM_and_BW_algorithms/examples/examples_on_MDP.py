import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from Estimation_algorithms_MDP import *
from examples_models import modelMDP2, scheduler_MDP2_random
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



m = modelMDP2()
s = scheduler_MDP2_random()

training_set_seq = []
training_set_val = []

for i in range(1000):
	trace = m.run(10, s)

	if not trace in training_set_seq:
		training_set_seq.append(res)
		training_set_val.append(0)

	training_set_val[training_set_seq.index(trace)] += 1

training_set = [training_set_seq,training_set_val]

algo = Estimation_algorithms_MDP(??,['tl','t','tr','l','n','r','bl','b','br'],['t','l','b','r'])
algo.problem3(training_set)