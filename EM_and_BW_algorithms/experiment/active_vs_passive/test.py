import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir1 = os.path.dirname(currentdir)
parentdir2 = os.path.dirname(parentdir1)
sys.path.append(parentdir2)

from examples.examples_models import modelMDP_bigstreet, scheduler_uniform
from examples.examples_runner import MDP, run_experiment
from src.tools import generateSet, loadSet, mergeSets
from src.models.MDP import loadMDP
from src.learning.Estimation_algorithm_MDP import Estimation_algorithm_MDP
import matplotlib.pyplot as plt

tr_size = 50
tr_len  = 6
nb_sta = 5
nb_seq = 2
nb_it = 50
output_folder = "/home/anna/Desktop/active_vs_passive_results"
kind_model = MDP

algorithm = "Active MDP-BW"


original = modelMDP_bigstreet()

training_set = generateSet(original,tr_size,tr_len,scheduler_uniform(original.actions()))

run_experiment(training_set,
			   output_folder,
			   kind_model,
			   algorithm,
			   model=original,
			   nb_states=nb_sta,
			   epsilon=0.01,
			   df=0.9,
			   lr=0,
			   nb_sequences=nb_seq,
			   nb_iteration=nb_it)


training_set = loadSet(output_folder+'/training_set.txt')
m = loadMDP(output_folder+"/model_0.txt")
algo = Estimation_algorithm_MDP(m,original.observations(),original.actions())
for i in range(1,nb_it+1):
	t = generateSet(original,nb_seq,tr_len,scheduler_uniform(original.actions()))
	training_set = mergeSets(training_set,t)
	algo.learn(training_set,output_file=output_folder+'/passive_models_'+str(i)+".txt",pp=str(i))

test_set = generateSet(original,200,tr_len,scheduler_uniform(original.actions()))

x = range(nb_it)

m = loadMDP(output_folder+"/model_0.txt")
y1 = [m.logLikelihood(test_set)]
y2 = [y1[0]]

for i in range(1,nb_it):
	m = loadMDP(output_folder+"/active_models_"+str(i)+".txt")
	y1.append(m.logLikelihood(test_set))
	m = loadMDP(output_folder+"/passive_models_"+str(i)+".txt")
	y2.append(m.logLikelihood(test_set))

fig, ax = plt.subplots()
ax.plot(x,y1,'r',label="Active  Learning")
ax.plot(x,y2,'g',label="Passive learning")
ax.legend()
plt.show()