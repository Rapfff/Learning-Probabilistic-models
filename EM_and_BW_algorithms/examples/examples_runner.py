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
model_to_learn = modelMCGT_game() #element of MDP,MCGT or HMM class
type_of_model = MCGT #here put HMM, MCGT or MDP

## *** training set ***
number_of_sequences = 100000 #int
length_of_each_sequence = 5 #int
#if we are learning an MDP (if not just set it to False)
schedulers = False #Should be a list (it can have one element) or False
fixed_action = False #bool

## *** initial model ***
number_of_states = 3 #int
###################################################

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

def chooseLearningAlgorithm(initial_model, type_of_model, fixed_action,observations, actions=None):
	if type_of_model == MDP and not fixed_action:
		return Estimation_algorithm_MDP_schedulers(initial_model,observations,actions)
	elif type_of_model == MDP and fixed_action:
		return Estimation_algorithm_MDP_sequences(initial_model,observations,actions)
	elif type_of_model == MCGT:
		return Estimation_algorithm_MCGT(initial_model,observations)
	elif type_of_model == HMM:
		return EM_ON_HMM(initial_model, observations)
	print("incorrect type_of_model value")

def runningExperiment(type_of_model,model_to_learn,number_of_sequences,length_of_each_sequence,schedulers,fixed_action,number_of_states):
	observations = model_to_learn.observations()
	if type_of_model == MDP:
		actions = model_to_learn.actions()
	else:
		actions = None

	training_set = generateTrainingSet(model_to_learn, number_of_sequences, length_of_each_sequence, schedulers, fixed_action)
	print(training_set)
	initial_model = generateRandomModel(type_of_model, number_of_states, observations, actions)
	initial_model.pprint()
	algo = chooseLearningAlgorithm(initial_model, type_of_model, fixed_action, observations, actions)

	final_loglikelihood, running_time = algo.problem3(training_set)
	output_model = algo.h
	output_model.pprint()
	print("Final loglikelihood:",final_loglikelihood)
	print("Running time:\t", running_time)
	return [final_loglikelihood,running_time,output_model]

def runningSeveralExperiments(number_experiments, output_file, type_of_model,model_to_learn,number_of_sequences,length_of_each_sequence,schedulers,fixed_action,number_of_states):
	output_file = open(output_file,'w')
	output_file.write("Model:\t"+str(model_to_learn)+"\n")
	output_file.write("Number of sequences:\t"+str(number_of_sequences)+"\n")
	output_file.write("Number of observations:\t"+str(length_of_each_sequence)+"\n")
	output_file.write("Fixed action:\t"+str(fixed_action)+"\n")
	output_file.write("Number of states:\t"+str(number_of_states)+"\n")
	output_file.write('\n')
	sum_loglikelihood = 0
	sum_running_time = 0
	best_loglikelihood = -256

	for i in range(number_experiments):
		final_loglikelihood,running_time,output_model = runningExperiment(type_of_model,model_to_learn,number_of_sequences,length_of_each_sequence,schedulers,fixed_action,number_of_states)
		output_file.write(str(final_loglikelihood)+"\n")
		output_model.save("smthing/")
		sum_loglikelihood += final_loglikelihood
		sum_running_time += running_time
		best_loglikelihood = max(best_loglikelihood,final_loglikelihood)
	
	output_file.write('\n')
	output_file.write("Average loglikelihood:\t"+str(sum_loglikelihood/number_experiments)+'\n')
	output_file.write("Best loglikelihood:\t"+str(best_loglikelihood)+'\n')
	output_file.write("Average running time:\t"+str(sum_running_time/number_experiments)+'\n')
	
	output_file.close()


###################################################

runningExperiment(type_of_model,model_to_learn,number_of_sequences,length_of_each_sequence,schedulers,fixed_action,number_of_states)
#runningSeveralExperiments(10,"test.txt",MDP,modelMDP5(),10,5,[scheduler_random(['a','b']),scheduler_always_same('a')],False,4)