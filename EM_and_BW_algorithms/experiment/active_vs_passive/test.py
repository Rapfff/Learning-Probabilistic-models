import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir1 = os.path.dirname(currentdir)
parentdir2 = os.path.dirname(parentdir1)
sys.path.append(parentdir2)

from examples.examples_models import modelMDP_bigstreet, scheduler_uniform, modelMDP_random
from examples.examples_runner import MDP, run_experiment
from src.tools import generateSet, loadSet, mergeSets, saveSet
from src.models.MDP import loadMDP
from src.learning.Estimation_algorithm_MDP import Estimation_algorithm_MDP
import matplotlib.pyplot as plt
from numpy.random import geometric


def folderFromParameters(output_folder,lr,epsilon_greedy,i):
	return output_folder+"/lr"+str(lr)+'_'+"epsilon"+str(epsilon_greedy)+"/exp"+str(i)


tr_size = 50
ts_size = 200
ts_len  = 6
nb_sta = 5
nb_seq = 1
nb_it = 200
output_folder = "/home/anna/Desktop/active_vs_passive_results/huge_experiment/big_street"
nb_steps = 6


learning_rates = [0]
epsilons_list = [round(0.1*i,2) for i in range(11)]
nb_exp = 20

original = modelMDP_bigstreet()

training_set = generateSet(original,tr_size,nb_steps,scheduler_uniform(original.actions()))
saveSet(training_set,output_folder+"/training_set.txt")


test_set = generateSet(original,ts_size,ts_len,scheduler_uniform(original.actions()))
saveSet(test_set,output_folder+"/test_set.txt")

m = modelMDP_random(nb_sta,original.observations(),original.actions())
m.save(output_folder+"/initial_model.txt")

algo = Estimation_algorithm_MDP(m,original.observations(),original.actions())
m = algo.learn(training_set,output_file=output_folder+"/model_0.txt")
m0_ll = str(m.logLikelihood(test_set))

for lr in learning_rates:
	for epsilon_greedy in epsilons_list:
		os.mkdir(output_folder+"/lr"+str(lr)+'_'+"epsilon"+str(epsilon_greedy))
		for i in range(1,nb_exp+1):
			os.mkdir(folderFromParameters(output_folder,lr,epsilon_greedy,i))
			m.save(folderFromParameters(output_folder,lr,epsilon_greedy,i)+"/model_0.txt")
		

fout = open(output_folder+"/csv_results.csv",'w')

for lr in learning_rates:
	for epsilon_greedy in epsilons_list:
		for i in range(1,nb_exp+1):
			training_set = loadSet(output_folder+"/training_set.txt")
			run_experiment(training_set,
						   folderFromParameters(output_folder,lr,epsilon_greedy,i),
						   MDP,
						   "Active MDP-BW",
						   model=original,
						   nb_states=nb_sta,
						   epsilon=0.01,
						   lr=lr,
						   epsilon_greedy=epsilon_greedy,
						   nb_sequences=nb_seq,
						   nb_iteration=nb_it,
						   nb_steps=nb_steps,
						   pp=str(i))
			results = "exp"+str(i)+','+str(lr)+','+str(epsilon_greedy)+','+m0_ll
			for j in range(nb_it):
				m = loadMDP(folderFromParameters(output_folder,lr,epsilon_greedy,i)+"/active_models_"+str(j+1)+".txt")
				results += ','+str(m.logLikelihood(test_set))
			results += '\n'
			fout.write(results)

fout.close()
"""
x = epsilons_list

m = loadMDP(output_folder+"/model_0.txt")
y1 = []
y2 = []

for epsilon_greedy in epsilons_list:
	m = loadMDP(output_folder+"/lrdynamic_"+"epsilon"+str(epsilon_greedy)+"/active_models_"+str(nb_it)+".txt")
	y1.append(m.logLikelihood(test_set))
for epsilon_greedy in epsilons_list:
	m = loadMDP(output_folder+"/lr0_"+"epsilon"+str(epsilon_greedy)+"/active_models_"+str(nb_it)+".txt")
	y2.append(m.logLikelihood(test_set))

plt.scatter(x,y1,c='r',label="Dynamic Learning Rate",alpha=0.5)
plt.scatter(x,y2,c='b',label="Learning on every seq",alpha=0.5)
plt.plot(x,y1,c='r',label="Dynamic Learning Rate",alpha=0.5)
plt.plot(x,y2,c='b',label="Learning on every seq",alpha=0.5)
plt.legend()
plt.show()
"""