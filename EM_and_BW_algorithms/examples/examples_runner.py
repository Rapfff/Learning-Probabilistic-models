import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

HMM = 0
MCGT = 1
MDP = 2

from Estimation_algorithms_MDP import *
from Estimation_algorithms_MCGT_multiple import *
from Estimation_algorithms_HMM import *
from tools import *
from alergia import *
from IOalergia import *
from examples_models import *
from MDP import *
from MCGT import *
from HMM import *


###################INPUTS##########################
model_to_learn = modelMDP3() #element of MDP,MCGT or HMM class
type_of_model = MDP #here put HMM, MCGT or MDP

## *** training set ***
number_of_sequences = 50 #int
length_of_each_sequence = 4 #int
#if we are learning an MDP (if not just set it to False)
schedulers = [scheduler_random(['a','b']),scheduler_random(['a','b'])] #Should be a list (it can have one element) or False
fixed_action = False #bool

## *** initial model ***
number_of_states = 4#int
###################################################

def generateSet(model,set_size,sequence_size,scheduler=None,with_action=False):
	seq = []
	val = []
	for i in range(set_size):
		if scheduler and with_action:
			trace = model.run(sequence_size,scheduler,True)
		elif scheduler:
			trace = model.run(sequence_size,scheduler,False)
		else:
			trace = model.run(sequence_size)

		if not trace in seq:
			seq.append(trace)
			val.append(0)

		val[seq.index(trace)] += 1

	return [seq,val]

def generateTrainingSet(model,set_size,sequence_size,scheduler,with_action):
	if not scheduler:
		return generateSet(model,set_size,sequence_size)
	elif len(scheduler) == 1 and with_action:
		return generateSet(model,set_size,sequence_size,scheduler[0],with_action)
	elif not with_action:
		res = []
		for s in scheduler:
			res.append([s,generateSet(model,set_size,sequence_size,s,with_action)])
		return res
	else:
		res = generateSet(model,set_size,sequence_size,scheduler[0],with_action)
		for s in scheduler[1:]:
			t = generateSet(model,set_size,sequence_size,s,with_action)
			for i in range(len(t[0])):
				if t[0][i] not in res[0]:
					res[0].append(t[0][i])
					res[1].append(t[1][i])
				else:
					res[1][res[0].index(t[0][i])] += t[1][i]
		return res

def generateRandomModel(type_of_model, number_of_states, observations, actions=None):
	if type_of_model == MDP:
		return modelMDP_random(number_of_states, observations, actions)
	elif type_of_model == MCGT:
		return modelMCGT_random(number_of_states, observations)
	elif type_of_model == HMM:
		return modelHMM_random(number_of_states, observations)
	print("incorrect type_of_model value")

def chooseLearningAlgorithm(initial_model, type_of_model, observations, actions=None):
	if type_of_model == MDP and not fixed_action:
		return Estimation_algorithm_MDP_schedulers(initial_model,observations,actions)
	elif type_of_model == MDP and fixed_action:
		return Estimation_algorithm_MDP_sequences(initial_model,observations,actions)
	elif type_of_model == MCGT:
		return Estimation_algorithm_MCGT(initial_model,observations)
	elif type_of_model == HMM:
		return EM_ON_HMM(initial_model, observations)
	print("incorrect type_of_model value")

###################################################
observations = model_to_learn.observations()
if type_of_model == MDP:
	actions = model_to_learn.actions()
else:
	actions = None

training_set = generateTrainingSet(model_to_learn, number_of_sequences, length_of_each_sequence, schedulers, fixed_action)
initial_model = generateRandomModel(type_of_model, number_of_states, observations, actions)
algo = chooseLearningAlgorithm(initial_model, type_of_model, observations, actions)

final_loglikelihood, running_time = algo.problem3(training_set)
output_model = algo.h

print("Running time:",running_time)
print("Final Loglikelihood:",final_loglikelihood)