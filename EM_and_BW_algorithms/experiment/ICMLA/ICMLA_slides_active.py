import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir1 = os.path.dirname(currentdir)
parentdir2 = os.path.dirname(parentdir1)
sys.path.append(parentdir2)

from src.tools import generateSet
from src.models.MDP import loadMDP
from examples.examples_models import scheduler_uniform, modelMDP5, modelMDP_random
from src.learning.Active_Learning_MDP import Active_Learning_MDP


original = modelMDP5()
sched = scheduler_uniform(original.actions())

training_set = generateSet(original,100,8,sched)

algo = Active_Learning_MDP(modelMDP_random(4,original.observations(),original.actions()),original.observations(),original.actions())
algo.learn(training_set,0,1,1)
