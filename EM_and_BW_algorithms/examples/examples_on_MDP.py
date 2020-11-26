import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from Estimation_algorithms_MDP import *
from examples_models import modelMDP1
from MDP import *

def scheduler_MDP1_a(path, current):
	if "a" in current.actions():
		return "a"
	else:
		return "b"

def scheduler_MDP1_b(path, current):
	if "b" in current.actions():
		return "b"
	else:
		return "a"

def scheduler_MDP1_change(path,current):
	actions = current.actions()
	if len(actions) == 1:
		return actions[0]
	if len(path) == 1:
		return "a"
	return actions[(actions.index(path[-2])+1) % len(actions)] #path[-2] = last action


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



m = modelMDP1()
fullyobservable_blackbox(m, 10, 100) 