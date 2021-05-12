import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from src.learning.Estimation_algorithm_MDP import *
from src.learning.Estimation_algorithm_MCGT import *
from src.learning.Active_Learning_MDP import Active_Learning_MDP
from src.tools import loadSet, generateSet, saveSet, getAlphabetFromSequences, getActionsObservationsFromSequences
from src.learning.alergia import *
from src.learning.IOalergia import *
from examples_models import *
from src.models.MDP import *
from src.models.MCGT import *
from time import time

MC = 2
MDP = 1

model = None
alpha = None
nb_states = None
limit = None
df = None
lr = None
nb_sequences = None
nb_iteration = None
test_set = None
#------------------------------------------------------------------------------


def run_experiment(training_set,output_folder,kind_model,algorithm,
				   test_set=None,model=None,alpha=None,nb_states=None,
				   limit=None,df=None,lr=None,nb_sequences=None,nb_iteration=None):
	saveSet(training_set,output_folder+"/training_set.txt")
	if test_set != None:
		saveSet(test_set,output_folder+"/test_set.txt")
	if model != None:
		model.save(output_folder+'/model_to_learn.txt')

	f = open(output_folder+"/parameters.txt",'w')
	f.write("Model to learn: ")
	if model != None:
		f.write(model.name+'\n')
	else:
		f.write("from training_set\n")
	f.write("Training_set: "+str(sum(training_set[1]))+" sequences of "+str(len(training_set[0][0]))+" labels\n")
	f.write("Algorithm: "+algorithm+'\n')
	if algorithm in ["IOAlergia","Alergia"]:
		f.write("alpha: "+str(alpha)+'\n')
	else:
		f.write("epsilon: "+str(limit)+'\n')
		if algorithm == "Active MDP-BW":
			f.write("discount factor: "+str(df)+'\n')
			f.write("learning rate: "+str(lr)+'\n')
			f.write("active learning sample size: "+str(nb_sequences)+'\n')
			f.write("active learning iterations: "+str(nb_iteration)+'\n')

	if model != None:
		observations = model.observations()
		if kind_model == MDP:
			actions = model.actions()
	else:
		if kind_model == MDP:
			actions, observations = getActionsObservationsFromSequences(training_set)
		else:
			observations = getAlphabetFromSequences(training_set)

	start_time = time()

	if algorithm == "Alergia":
		algo = Alergia()
		output_model = algo.learn(training_set,alpha,observations)
		output_model.save(output_folder+'/output_model.txt')

	elif algorithm =="IOAlergia":
		algo = IOAlergia()
		output_model = algo.learn(training_set,alpha,actions,observations)
		output_model.save(output_folder+'/output_model.txt')

	elif algorithm == "MC-BW":
		initial_model = modelMCGT_random(nb_states,observations)
		initial_model.save(output_folder+'/initial_model.txt')
		algo = Estimation_algorithm_MCGT(initial_model,observations)
		output_model = algo.learn(training_set,output_folder+'/output_model.txt',limit)

	else:
		initial_model = modelMDP_random(nb_states,observations,actions)
		initial_model.save(output_folder+'/initial_model.txt')

		if algorithm == "Passive MDP-BW":
			algo = Estimation_algorithm_MDP(initial_model,observations,actions)
			output_model = algo.learn(training_set,output_folder+'/output_model.txt',limit)
		else:
			algo = Active_Learning_MDP(initial_model,observations,actions)
			output_model = algo.learn(training_set,df,lr,nb_sequences,nb_iteration,output_folder,limit)

	delta_time = time()-start_time
	f.write("number of states in the output model: "+str(len(output_model.states))+'\n')

	f.write("Learning duration: "+str(delta_time)+'\n')
	
	f.write("Loglikelihood on the training set: "+str(output_model.logLikelihood(training_set))+'\n')
	if test_set != None:
		f.write("Loglikelihood on the test set: "+str(output_model.logLikelihood(test_set))+'\n')
	f.close()

	print("Results written in "+output_folder+"/parameters.txt")



if __name__ == '__main__':
	print("(1/17) Which kind of model do you want to learn? (1: MDP, 2: MC)")
	kind_model = int(input())
	while kind_model not in [1,2]:
		kind_model = int(input())

	print("(2/17) Do you already have the training set or do you want me to generate it? (y: yes, I have the training set, n: no I don't, please generate it)")
	answer = input()
	while answer not in ['y','n']:
		answer = input()

	if answer == 'y':
		print("(3/17) Enter the location of the training set file:")
		training_set = loadSet(input())
		model = None
		print("Questions 4, 5 and 6 skipped...")
	else:
		print("(3/17) Do you want to learn a model saved in a file or to choose one from the example list? (y: yes, from a file, n: no, I would like to choose from the list)")
		answer = input()
		while answer not in ['y','n']:
			answer = input()
		if answer == 'y':
			print("(4/17) Enter the location of the model file:\n")
			if kind_model == MDP:
				model = loadMDP(answer)
			else:
				model = loadMCGT(answer)
		else:
			print("(4/17) You can choose between:")
			if kind_model == MC:
				l = ["MC 1          (states: 5, labels: 7)",
					 "MC 2          (states: 4, labels: 5)",
					 "MC 3          (states: 5, labels: 5)",
					 "MC 4          (states: 6, labels: 7)",
					 "small game    (states: 3, labels: 6)",
					 "REBER grammar (states: 7, labels:7)"]
				for i in range(len(l)):
					print(i+1," -",l[i])
				answer = int(input())
				while answer not in range(1,len(l)+1):
					answer = int(input())
				if answer == 1:
					model = modelMCGT1()
				elif answer == 2:
					model = modelMCGT2()
				elif answer == 3:
					model = modelMCGT3()
				elif answer == 4:
					model = modelMCGT4()
				elif answer == 5:
					model = modelMCGT_game()
				elif answer == 6:
					model = modelMCGT_REBER()
			else:
				l = ["MDP 1        (states: 2, actions: 2, labels: 2)",
					 "MDP 2        (states: 3, actions: 2, labels: 3)",
					 "MDP 3        (states: 4, actions: 2, labels: 3)",
					 "small street (states: 2, labels: 3)",
					 "mid street   (states: 4, labels: 3)",
					 "big street   (states: 5, labels: 4)"]
				for i in range(len(l)):
					print(i+1," -",l[i])
				answer = int(input())
				while answer not in range(1,len(l)+1):
					answer = int(input())
				if answer == 1:
					model = modelMDP3()
				elif answer == 2:
					model = modelMDP4()
				elif answer == 3:
					model = modelMDP5()
				elif answer == 4:
					model = modelMDP_smallstreet()
				elif answer == 5:
					model = modelMDP_midstreet()
				elif answer == 6:
					model = modelMDP_bigstreet()


		print("(5/17) How many sequences do you want in the training set?")
		size_training_set = int(input())
		print("(6/17) How many labels should contains each sequence?")
		len_training_set = int(input())
		print("Generating the training set...")
		if kind_model == MDP:
			print("(Note: since your learning an MDP I will generate the sequences using a uniform scheduler)")
			training_set = generateSet(model,size_training_set,len_training_set,scheduler_uniform(model.actions()))
		else:
			training_set = generateSet(model,size_training_set,len_training_set)

	print("(7/17) Which learning algorithm do you want to use? ",end="")
	if kind_model == MC:
		print("(1: MC-BW, 2: Alergia)")
		answer = int(input())
		while answer not in range(1,3):
			answer = int(input())
		algorithm = ['MC-BW','Alergia'][answer-1]

	else:
		print("(1: Active MDP-BW, 2: Passive MDP-BW, 3: IOAlergia)")
		answer = int(input())
		while answer not in range(1,4):
			answer = int(input())
		algorithm = ['Active MDP-BW','Passive MDP-BW','IOAlergia'][answer-1]


	if algorithm in ["IOAlergia","Alergia"]:
		print("(8/17) Enter the value for the alpha parameter (between 0 and 1):")
		alpha = float(input())
		print("Questions 9, 10, 11, 12 and 13 skipped...")
	else:
		print("(8/17) Enter the number of states in the output model:")
		nb_states = int(input())	
		print("(9/17) Enter the value for the epsilon parameter (default is 0.01):")
		limit = float(input())
		if algorithm == "Active MDP-BW":
			print("(10/17) Enter the value for the discount factor (default 0.9):")
			df = float(input())
			print("(11/17) Enter the value for the learning rate ('dynamic', 0, or a number between 0 and 1):")
			answer = input()
			if answer == '0':
				lr = 0
			elif answer == 'dynamic':
				lr = 'dynamic'
			else:
				lr = float(answer)
			print("(12/17) Enter the size of the sample for each active learning iteration:")
			nb_sequences = int(input())
			print("(13/17) Enter the number of active learning iterations:")
			nb_iteration = int(input())
		else:
			print("Questions 10, 11, 12 and 13 skipped...")

	print("(14/17) Do you have a test set? (y: yes, g: no but I want you to generate one, n: no and I don't want to have one)")
	answer = input()
	while answer not in ['y','g','n']:
		answer = input()
	if answer == 'y':
		print("(15/17) Enter the location of your test set file:")
		test_set = loadSet(input())
		print("Question 16 skipped...")
	elif answer == 'g' and model == None:
		print("I cannot generate a test set since I don't have the original model...")
	elif answer == 'g' and model != None:
		print("(15/17) Enter the size of the test set:")
		size_test_set = int(input())
		print("(16/17) How many labels should contains each sequence?")
		len_test_set = int(input())
		print("Generating the test set...")
		if kind_model == MDP:
			print("(Note: since your learning an MDP I will generate the sequences using a uniform scheduler)")
			test_set = generateSet(model,size_test_set,len_test_set,scheduler_uniform(model.actions()))
		else:
			test_set = generateSet(model,size_test_set,len_test_set)
	else:
		print("Questions 15 and 16 skipped...")

	print("(17/17) Finally enter the output folder (it should be already created):")
	output_folder = input()


	run_experiment(training_set,output_folder,kind_model,algorithm,test_set,model,alpha,nb_states,limit,df,lr,nb_sequences,nb_iteration)