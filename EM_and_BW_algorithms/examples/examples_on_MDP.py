import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from Estimation_algorithms_MDP import *
from examples_models import modelMDP1
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

