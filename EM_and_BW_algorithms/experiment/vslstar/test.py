import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir1 = os.path.dirname(currentdir)
parentdir2 = os.path.dirname(parentdir1)
sys.path.append(parentdir2)

from gridMDP import Grid
from examples.examples_models import scheduler_uniform, modelMDP_random
from src.tools import generateSet, loadSet, saveSet
from src.models.MDP import loadMDP
from numpy.random import geometric

tr_size = 200
tr_len = 0.025
ts_size = 200
ts_len  = 50
nb_sta = 35
nb_seq = 2
nb_it = 1500

min_size = 10

nb_steps = []
for i in range(nb_it):
	nb_steps.append([])
	for j in range(nb_seq):
		nb_steps[-1].append(min_size+geometric(tr_len))
print(nb_steps)

original = Grid()

#training_set = generateSet(original,tr_size,tr_len,scheduler_uniform(original.actions()),distribution='geo',min_size=min_size)
#saveSet(training_set,"training_set.txt")
training_set = loadSet(output_folder+"/training_set.txt")

#m = modelMDP_random(nb_sta,original.observations(),original.actions())
#m.save("initial_model.txt")
m = loadMDP("initial_model.txt")

algo = Active_Learning_MDP(m, original.observations(), original.actions() )
algo.learn(training_set,0,nb_seq,nb_it,epsilon_greedy=1.0,epsilon=0.01,number_steps=nb_steps,pp='')
