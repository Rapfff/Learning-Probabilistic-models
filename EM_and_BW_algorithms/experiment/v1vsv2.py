import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)sys.path.append(parentdir)

from src.tools import saveSet, loadSet, generateSet
from src.models.MDP import loadMDP
from src.learning.Estimation_algorithm_MDP import Estimation_algorithm_MDP
from src.learning.Estimation_algorithm_MDPv2 import Estimation_algorithm_MDPv2
from examples.examples_models import scheduler_uniform,modelMDP_random,modelMDP5

alphabet = ['A','B','C']
actions = ['a','b']

initial_model = modelMDP_random(4,alphabet,actions)
initial_model.save("v1vsv2/initial_model.txt")

original_model = modelMDP5()
scheduler = scheduler_uniform(actions)
training_set = generateSet(original_model,1000,5,scheduler)
saveSet(training_set,"v1vsv2/training_set.txt")

initial_model = loadMDP("v1vsv2/initial_model.txt")
training_set = loadSet("v1vsv2/training_set.txt")

algo = Estimation_algorithm_MDP(initial_model,alphabet,actions)
algo.learn(training_set,output_file="v1vsv2/output_modelv1.txt")


algo = Estimation_algorithm_MDPv2(initial_model,alphabet,actions)
algo.learn(training_set,output_file="v1vsv2/output_modelv2.txt")
