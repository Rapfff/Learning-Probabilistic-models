import os, sys, copy
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from EM_and_BW_algorithms.examples.examples_models import *
from EM_and_BW_algorithms.src.learning.Estimation_algorithm_HMM import *
from Experiment import *

def modelMCGT_equiprobable(nb_states,alphabet):
	s = []
	for i in range(nb_states):
		s += [i] * len(alphabet)
	obs = alphabet*nb_states
	
	states = []
	for i in range(nb_states):
		states.append(MCGT_state([[1/len(obs)]*len(obs),s,obs]))
	return MCGT(states,0,"MCGT_equiprobable_"+str(nb_states)+"states")

def modelMCGT_random(nb_states,alphabet):
	s = []
	for i in range(nb_states):
		s += [i] * len(alphabet)
	obs = alphabet*nb_states
	
	states = []
	for i in range(nb_states):
		states.append(MCGT_state([randomProbabilities(len(obs)),s,obs]))
	return MCGT(states,0,"MCGT_random_"+str(nb_states)+"states")


#def modelMCGT_rectangle( width, height, alphabet ):
#	s = []
#    nb_states = width*height+2
#	for i in range(nb_states):
#		s += [i] * len(alphabet)
#	#obs = alphabet*nb_states
#	
#	states = []
#    # initial state, connects to each of first column
#    sates.append(MCGT_state( [[1/height]*height,s[1:height+1], alphabet* ] ))
#
#	for i in range(nb_states):
#		states.append(MCGT_state( [ [1/len(obs)]*len(obs), s, obs ] ))
#	return MCGT(states,0,"MCGT_equiprobable_"+str(nb_states)+"states")

m = modelMCGT_equiprobable( 4, "abcd" )
print( m.states[0].next_matrix )
