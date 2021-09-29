import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)sys.path.append(parentdir)

from src.tools import saveSet, loadSet, generateSet
from src.models.MDP import loadMDP
from src.learning.Estimation_algorithm_MCGT import Estimation_algorithm_MCGT
from src.learning.Estimation_algorithm_MCGTv2 import Estimation_algorithm_MCGTv2
from examples.examples_models import scheduler_uniform, modelMCGT_random, modelMCGT_REBER

alphabet = ['B','E','P','S','T','V','X']

initial_model = modelMCGT_random(7,alphabet)
initial_model.save("v1vsv2/initial_model.txt")

original_model = modelMCGT_REBER()
training_set = generateSet(original_model, 1000, 5)
saveSet(training_set,"v1vsv2/training_set.txt")

initial_model = loadMCGT("v1vsv2/initial_model.txt")
training_set = loadSet("v1vsv2/training_set.txt")

algo = Estimation_algorithm_MCGT(initial_model,alphabet)
algo.learn(training_set,output_file="v1vsv2/output_modelv1.txt").pprint()

algo = Estimation_algorithm_MCGTv2(initial_model,alphabet)
algo.learn(training_set,output_file="v1vsv2/output_modelv2.txt").pprint()